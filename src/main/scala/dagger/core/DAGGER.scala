package dagger.core

import java.io.FileWriter

import dagger.ml._
import scala.collection.parallel.{ ParIterable, ForkJoinTaskSupport }
import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.forkjoin.ForkJoinPool
import scala.reflect.ClassTag
import scala.util.Random
import gnu.trove.map.hash.THashMap

/**
 * Created by narad on 6/10/14.
 */
// Dagger[Action, State]
// D = Data
// A = Action
// S = State
class DAGGER[D: ClassTag, A <: TransitionAction[S]: ClassTag, S <: TransitionState: ClassTag](options: DAGGEROptions) {
  val random = new Random(options.RANDOM_SEED)

  def train(data: Iterable[D],
    expert: HeuristicPolicy[D, A, S],
    featureFactory: FeatureFunctionFactory[D, S, A],
    trans: TransitionSystem[D, A, S],
    lossFactory: LossFunctionFactory[D, A, S],
    dev: Iterable[D] = Iterable.empty,
    score: Iterable[(D, D)] => List[(String, Double)],
    actionToString: (A => String) = null,
    stringToAction: (String => A) = null,
    utilityFunction: (DAGGEROptions, String, Integer, D, D) => Unit = null,
    startingClassifier: MultiClassClassifier[A] = null): MultiClassClassifier[A] = {
    // Construct new classifier and uniform classifier policy
    //    val dataSize = data.size
    //    val cache = new mutable.HashMap[S, Array[Double]]

    //  def features = (d: D, s: S, a: A) => Map[Int, Double]()
    // Begin DAGGER training
    var instances = new ArrayBuffer[Instance[A]]
    var classifier = startingClassifier
    var policy = new ProbabilisticClassifierPolicy[D, A, S](classifier)
    val initialOracleLoss = options.ORACLE_LOSS
    for (i <- 1 to options.DAGGER_ITERATIONS) {
      if (i == 1 && options.INITIAL_ORACLE_LOSS) options.ORACLE_LOSS = true else options.ORACLE_LOSS = initialOracleLoss
      if (i > 1) options.USE_EXPERT_ON_ROLLOUT_AFTER += options.EXPERT_HORIZON_INCREMENT
      val prob = if (i == 1) 1.0 else options.INITIAL_EXPERT_PROB * math.pow(1.0 - options.POLICY_DECAY, i - 1)
      println("DAGGER iteration %d of %d with P(oracle) = %.2f".format(i, options.DAGGER_ITERATIONS, prob))
      val newInstances = collectInstances(data, expert, policy, featureFactory, trans, lossFactory, prob, i, utilityFunction)
      if (actionToString != null) {
        writeInstancesToFile(newInstances, i, actionToString)
      }
      if (stringToAction == null) {
        val starting = instances.size
        instances = instances filter (i => i.getErrorCount < options.INSTANCE_ERROR_MAX)
        println(starting - instances.size + " instances dropped for exceeding error threshold.")
        instances ++= newInstances
        println("DAGGER iteration - training classifier on " + instances.size + " total instances.")
        if (options.PLOT_LOSS_PER_ITERATION) {
          val totalIterations = options.TRAIN_ITERATIONS
          options.TRAIN_ITERATIONS = 1
          for (j <- 1 to totalIterations) {
            classifier = trainFromInstances(instances, trans.actions, old = classifier)
            if (dev.nonEmpty) stats(data, j, dev, new ProbabilisticClassifierPolicy[D, A, S](classifier), trans,
              featureFactory.newFeatureFunction.features, lossFactory, score, utilityFunction)
          }
          options.TRAIN_ITERATIONS = totalIterations
        } else {
          classifier = trainFromInstances(instances, trans.actions, old = classifier)
        }

      } else {
        // load from file - FileInstances is an Iterable that only loads each file as needed
        val startingInstances = math.max(1, i - options.PREVIOUS_ITERATIONS_TO_USE)
        println("Starting from " + startingInstances)
        val fileNames = ((startingInstances to i) map (iter => options.DAGGER_OUTPUT_PATH + "Instances_" + iter + ".txt")).toList
        classifier = trainFromInstances(new FileInstances(fileNames, stringToAction, actionToString, options.INSTANCE_ERROR_MAX), trans.actions, old = classifier)
      }

      policy = new ProbabilisticClassifierPolicy[D, A, S](classifier)
      // Optionally discard old training instances, as in pure imitation learning
      if (options.DISCARD_OLD_INSTANCES) instances.clear()
      if (dev.nonEmpty && !options.PLOT_LOSS_PER_ITERATION) stats(data, i, dev, policy, trans, featureFactory.newFeatureFunction.features, lossFactory, score, utilityFunction)
    }
    classifier
  }

  def writeInstancesToFile(instances: Array[Instance[A]], iteration: Int, actionToString: (A => String)) {
    val fileName = options.DAGGER_OUTPUT_PATH + "Instances_" + iteration + ".txt"
    val file = new FileWriter(fileName)
    for (i <- instances) {
      file.write(i.fileFormat(actionToString))
    }
    file.close
  }

  def collectInstances(data: Iterable[D], expert: HeuristicPolicy[D, A, S], policy: ProbabilisticClassifierPolicy[D, A, S],
    featureFactory: FeatureFunctionFactory[D, S, A], trans: TransitionSystem[D, A, S], lossFactory: LossFunctionFactory[D, A, S],
    prob: Double = 1.0, iteration: Int, utilityFunction: (DAGGEROptions, String, Integer, D, D) => Unit): Array[Instance[A]] = {
    val timer = new dagger.util.Timer
    timer.start()

    val Dagger = options.ALGORITHM == "Dagger" || options.ALGORITHM == "DILO" || options.ALGORITHM == "DILDO"
    val LOLSDet = options.ALGORITHM == "LOLSDet" || options.ALGORITHM == "DILDO"
    val LOLS = options.ALGORITHM == "LOLS" || options.ALGORITHM == "DILO" || options.ALGORITHM == "LIDO"
    // Compute the probability of choosing the oracle policy in this round
    //      val prob = math.pow(1.0 - options.POLICY_DECAY, i-1)
    //      println("DAGGER iteration %d of %d with P(oracle) = %.2f".format(i, options.DAGGER_ITERATIONS, prob))
    // Keep statistics on # of failed unrolls and the accuracy of predicted structures
    //    var numFailedUnrolls = 0
    //    var numCorrectUnrolls = 0
    val debug = if (options.DEBUG) new FileWriter(options.DAGGER_OUTPUT_PATH + "CollectInstances_debug_" + iteration + "_" + f"$prob%.3f" + ".txt") else null
    var lossOnTestSet = List[Double]()
    var processedSoFar = 0
    val dataWithIndex = data.zipWithIndex
    val MAX_ACTIONS = options.MAX_ACTIONS

    val allData = fork(dataWithIndex, options.NUM_CORES).flatMap {
      case (d, dcount) =>
        val instances = new ArrayBuffer[Instance[A]]
        // We create new Loss and Feature functions each time for thread-safety as they cache some results for performance reasons
        val loss = lossFactory.newLossFunction
        val featFn = featureFactory.newFeatureFunction

        // Use policies to fully construct (unroll) instance from start state
        val (_, expertActions, _) = if (options.UNROLL_EXPERT_FOR_LOSS) unroll(d, expert, policy, trans.init(d), trans, featFn.features, 1.0) else (0, Array[A](), 0)
        val (predEx, predActions, expertUse) = unroll(d, expert, policy, trans.init(d), trans, featFn.features, prob, true)
        if (options.DEBUG) {
          debug.write("Initial State:" + trans.init(d) + "\n")
          debug.write("Actions Taken:\n")
          (predActions zip expertUse) foreach (x => debug.write(x._1 + " : " + x._2 + "\n"))
          //     debug.write("Final State: " + predEx + "\n")
        }

        val totalLoss = predEx match {
          case None => 1.0
          case Some(output) =>
            if (utilityFunction != null) utilityFunction(options, "instanceCollection_" + iteration.toString, dcount + 1, output, d)
            loss(output, d, predActions, expertActions)
        }
        this.synchronized {
          lossOnTestSet = totalLoss :: lossOnTestSet
        }
        if (options.DEBUG) debug.write(f"Total Loss:\t$totalLoss%.3f, using ${predActions.size}\n")
        // Check that the oracle policy is correct
        //        if (i == 1 && options.CHECK_ORACLE) assert(predEx.get == d, "Oracle failed to produce gold structure...Gold:\n%s\nPredicted:\n%s".format(d, predEx.get))
        //        if (predEx.get == d) numCorrectUnrolls += 1
        // Initialize state from data example
        var state = trans.init(d)
        // For all actions used to predict the unrolled structure...
        //for (a <- predActions)
        loss.setSamples(options.NUM_SAMPLES * (if (LOLSDet && policy.classifier != null && !options.ORACLE_LOSS) 2 else 1))
        val allInstances = predActions.zipWithIndex map {
          case (a, actionNumber) =>

            // Find all actions permissible for current state
            val allPermissibleActions = trans.permissibleActions(state)
            val nextExpertAction = expert.chooseTransition(d, state)
            val nextPolicyActionsAndScores = if (policy.classifier != null)
              predictUsingPolicy(d, state, policy, allPermissibleActions, featFn.features, options.ROLLOUT_THRESHOLD)
            else {
              // pick a non-expert action at random
              val excludingExpertChoice = allPermissibleActions.filterNot { x => x == nextExpertAction }
              val allChoices = (if (excludingExpertChoice.size > 0) Array(excludingExpertChoice(Random.nextInt(excludingExpertChoice.size))) else Array(nextExpertAction)).toSeq
              allChoices map { x => (x, 0.0f) }
            }
            if (options.DEBUG && options.ROLLOUT_THRESHOLD > 0.0) {
              debug.write("\n" + (nextPolicyActionsAndScores sortWith { case ((a1: A, s1: Float), (a2: A, s2: Float)) => s1 > s2 } map { case (action, score) => f"$action $score%.3f\t" }).mkString(" ") + "\n")
            }
            val nextPolicyActions = nextPolicyActionsAndScores map { x => x._1 }
            val policyActionScores = nextPolicyActionsAndScores.toMap
            val permissibleActions = options.REDUCED_ACTION_SPACE match {
              case true if (nextPolicyActions contains nextExpertAction) => nextPolicyActions.toArray
              case true => (nextPolicyActions ++ Seq(nextExpertAction)).toArray
              case false if (allPermissibleActions contains nextExpertAction) => allPermissibleActions
              case false => allPermissibleActions :+ nextExpertAction
            }

            if (options.DEBUG) {
              if (true) debug.write("State:" + state + "\n")
              debug.write("\n" + (permissibleActions.mkString("; ") + "\n"))
              debug.flush
            }
            // Compute a cost for each permissible action
            // if just one action, then we can save some time in not rolling out
            val costs = if (permissibleActions.size == 1) Array(0.0) else permissibleActions.map { l =>

              def calculateAndLogLoss(ex: Option[D], actions: Array[A], expert: Array[Boolean], expertActionsFromHere: Array[A], lastAction: A, nextExpertAction: A): Double = {
                if (options.ORACLE_LOSS) return if (l == nextExpertAction) 0.0 else 1.0
                (ex, if (expert.length > 0) expert(0) else false) match {
                  case (None, _) =>
                    if (options.DEBUG) debug.write("Failed unroll, loss = " + loss.max(d) + "\n")
                    loss.max(d)
                  case (Some(structure), usedExpert) =>
                    val basicLoss = loss(gold = d, test = structure, actions, expertActionsFromHere, lastAction, nextExpertAction)
                    val pScore = policyActionScores.getOrElse(lastAction, -1.0f)
                    val fullLoss = if (options.COACHING_LAMBDA > 0.0 && policy.classifier != null) basicLoss + options.COACHING_LAMBDA * pScore else basicLoss
                    if (options.DEBUG) {
                      debug.write(f"Loss on action $lastAction = $basicLoss%.3f; Policy score $pScore%.3f; Full Loss $fullLoss%.3f (${if (usedExpert) "Expert" else "Learned Policy"})\n")
                      if (true) {
                        actions foreach { a => debug.write(a.toString + "\t") }
                        debug.write("\n")
                      }
                    }
                    fullLoss
                }
              }

              (1 to options.NUM_SAMPLES).map { s =>
                // Create a copy of the state and apply the action for the cost calculation
                var stateCopy = state
                stateCopy = l(stateCopy)
                //        if (options.EXPERT_APPROXIMATION) {
                //          loss(gold = d, test = trans.expertApprox(d, stateCopy), testActions = Array(), expertActions)
                //        }
                options.MAX_ACTIONS = MAX_ACTIONS - actionNumber
                val (_, expertActionsFromHere, _) = if (options.UNROLL_EXPERT_FOR_LOSS) unroll(d, expert, policy, stateCopy, trans, featFn.features, 1.0) else (0, Array[A](), 0)
                if (options.APPROXIMATE_LOSS) {
                  trans.approximateLoss(datum = d, state = state, action = l)
                } else if (LOLSDet && policy.classifier != null && !options.ORACLE_LOSS) {
                  val (sampledExEx, sampledActionsEx, _) = unroll(d, expert, policy, stateCopy, trans, featFn.features, 1.0) // uses expert
                  val expertLoss = calculateAndLogLoss(sampledExEx, sampledActionsEx, Array(true), expertActionsFromHere, l, nextExpertAction)
                  val (sampledExLP, sampledActionsLP, _) = unroll(d, expert, policy, stateCopy, trans, featFn.features, 0.0) // uses learned policy
                  val policyLoss = calculateAndLogLoss(sampledExLP, sampledActionsLP, Array(false), expertActionsFromHere, l, nextExpertAction)
                  expertLoss * prob + policyLoss * (1.0 - prob)
                } else {
                  // Unroll from current state until completion
                  val (sampledEx, sampledActions, expertInSample) = if (!options.ORACLE_LOSS) unroll(d, expert, policy, stateCopy, trans, featFn.features, prob) else (Some(d), Array[A](), Array[Boolean]())
                  calculateAndLogLoss(sampledEx, sampledActions, expertInSample, expertActionsFromHere, l, nextExpertAction)
                }
              }.foldLeft(0.0)(_ + _) / options.NUM_SAMPLES // average the label loss for all samples
            }
            // Reduce all costs until the min cost is 0

            val coachedCosts = costs
            val min = costs.minBy(_ * 1.0)
            val costsWithoutOffset = costs.map(x => (x - min))
            val normedCosts = if (options.BINARY_LOSS) costsWithoutOffset map (x => if (x == 0.0) 0.0 else 1.0) else costsWithoutOffset
            if (options.DEBUG) {
              debug.write("Actions = " + permissibleActions.mkString(", ") + "\n")
              debug.write("Original Costs = " + (costs map (i => f"$i%.3f")).mkString(", ") + "\n")
              debug.write("Normed Costs = " + (normedCosts map (i => f"$i%.3f")).mkString(", ") + "\n")

              val minAction = permissibleActions(normedCosts.indexOf(0.0))
              debug.write("Expert action: " + nextExpertAction + ", versus min cost action: " + minAction + "\n")
              if (!(permissibleActions contains nextExpertAction)) debug.write("Expert Action is not in permissible set.")
              debug.write("\n")
              debug.flush()
            }
            // Construct new training instance with sampled losses
            //TODO: We're keeping too many copies of features. We can dump those that are identical
            val allFeatures = permissibleActions map (a => featFn.features(d, state, a))
            val weightLabels = permissibleActions map (_.getMasterLabel.asInstanceOf[A])
            val instance = new Instance[A](allFeatures.toList, permissibleActions, weightLabels, normedCosts map (_.toFloat))
            //  println(f"${permissibleActions.mkString(",")}, maxCost ${instance.maxCost}%.2f, minCost ${instance.minCost}%.2f, correctLabels ${instance.correctLabels}")
            loss.clearCache
            //        if (options.SERIALIZE) file.write(instance.toSerialString + "\n\n") else instances += instance

            // Progress to next state in the predicted path
            state = a(state)
            instance
        }
        options.MAX_ACTIONS = MAX_ACTIONS
        this.synchronized {
          processedSoFar += 1
          if (processedSoFar % options.DAGGER_PRINT_INTERVAL == 0) {
            System.err.print("\r..instance %d in %s, average time per instance = %s".format(dcount + 1, timer.toString, timer.toString(divisor = processedSoFar)))
          }
        }
        allInstances
    }.toArray
    if (options.DEBUG) debug.write(f"Mean Loss on test set:\t ${(lossOnTestSet reduce (_ + _)) / lossOnTestSet.size}%.3f")
    if (options.DEBUG) debug.close
    allData filter (_.costs.size > 1)
  }

  /**
   * A function which uses the transition system and parameters to construct an output of type D
   * beginning from the provided start state.
   * @param ex the gold instance, used by the oracle / heuristic policy (expert policy)
   * @param expertPolicy the oracle / heuristic policy
   * @param classifierPolicy the probabilistic policy
   * @param start the start state (which may represent a partial unrolling)
   * @param trans a transition system for transitioning from one state to another
   * @param featureFunction a feature function used in constructing a new instance
   * @param prob the probability of selecting the oracle policy
   */
  def unroll(ex: D,
    expertPolicy: HeuristicPolicy[D, A, S],
    classifierPolicy: ProbabilisticClassifierPolicy[D, A, S],
    start: S, trans: TransitionSystem[D, A, S],
    featureFunction: (D, S, A) => gnu.trove.map.hash.THashMap[Int, Float],
    probability: Double = 1.0, rollIn: Boolean = false): (Option[D], Array[A], Array[Boolean]) = {

    // For Dagger we have no difference between Roll-In and Roll-Out. We always use expert with probability.
    // For LOLS we always Roll-In with the learned policy (if we have one, which we won't on the first iteration)
    val prob = (options.ALGORITHM, rollIn) match {
      case ("LOLS", true) | ("LOLSDet", true) | ("LIDO", true) => if (classifierPolicy.classifier == null) 1.0 else 0.0
      case (_, true) => probability
      case ("Dagger", false) | ("LIDO", false) => probability
      case (_, false) => if (classifierPolicy.classifier == null || Random.nextDouble < probability) 1.0 else 0.0
    }
    val actions = new ArrayBuffer[A]
    val expertUsed = new ArrayBuffer[Boolean]
    var state = start
    var actionsTaken = 0
    while (!trans.isTerminal(state) && actionsTaken < options.MAX_ACTIONS) {
      val permissibleActions = trans.permissibleActions(state)
      if (permissibleActions.isEmpty) {
        return (None, actions.toArray, expertUsed.toArray)
      }
      val policy = (rollIn, actionsTaken >= options.USE_EXPERT_ON_ROLLOUT_AFTER, (classifierPolicy.classifier == null || random.nextDouble <= prob)) match {
        case (true, _, true) =>
          expertUsed += true; expertPolicy
        case (true, _, false) =>
          expertUsed += false; classifierPolicy
        case (false, false, false) =>
          expertUsed += false; classifierPolicy
        case (false, false, true) =>
          expertUsed += true; expertPolicy
        case (false, true, _) =>
          expertUsed += true; expertPolicy
      }
      val a = policy match {
        case x: HeuristicPolicy[D, A, S] => x.predict(ex, state)
        case y: ProbabilisticClassifierPolicy[D, A, S] => predictUsingPolicy(ex, state, y, permissibleActions, featureFunction, 0.0).head._1
      }

      actions += a
      actionsTaken += 1
      if (actionsTaken == options.MAX_ACTIONS && options.DEBUG) {
        println(s"Unroll terminated at $actionsTaken actions")
      }
      state = a(state)
    }
    (Some(trans.construct(state, ex)), actions.toArray, expertUsed.toArray)
  }

  def predictUsingPolicy(ex: D, state: S, policy: ProbabilisticClassifierPolicy[D, A, S], permissibleActions: Array[A],
    featureFunction: (D, S, A) => THashMap[Int, Float], threshold: Double): Seq[(A, Float)] = {
    val weightLabels = permissibleActions map (_.getMasterLabel.asInstanceOf[A])
    val instance = new Instance[A]((permissibleActions map (a => featureFunction(ex, state, a))).toList,
      permissibleActions, weightLabels, permissibleActions.map(_ => 0.0f))
    val prediction = policy.predict(ex, instance, state, threshold)
    prediction
  }

  def trainFromInstances(instances: Iterable[Instance[A]], actions: Array[A], old: MultiClassClassifier[A]): MultiClassClassifier[A] = {
    val weightLabels = actions map (_.getMasterLabel.asInstanceOf[A])
    options.CLASSIFIER match {
      case "AROW" => {
        old match {
          case c: AROWClassifier[A] => AROW.train[A](instances, weightLabels, options, Some(c))
          case _ => AROW.train[A](instances, weightLabels, options)
        }
      }
      case "PASSIVE_AGGRESSIVE" => ??? //PassiveAggressive.train[A](instances, actions, options.RATE, random, options)
      case "PERCEPTRON" => ??? // Perceptron.train[A](instances, actions, options.RATE, random, options)
    }
  }
  def decode(ex: D, classifierPolicy: ProbabilisticClassifierPolicy[D, A, S],
    trans: TransitionSystem[D, A, S], featureFunction: (D, S, A) => gnu.trove.map.hash.THashMap[Int, Float]): (Option[D], Array[A]) = {
    unroll(ex, expertPolicy = null, classifierPolicy, start = trans.init(ex), trans, featureFunction, probability = 0.0, true) match { case (a, b, c) => (a, b) }
  }

  def fork[T](data: Iterable[T], forkSize: Int): ParIterable[T] = {
    System.err.println("Parallelizing to %d cores...".format(forkSize))
    val par = data.par
    par.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(forkSize))
    par
  }

  def stats(trainingData: Iterable[D], iter: Integer, validationData: Iterable[D], policy: ProbabilisticClassifierPolicy[D, A, S], trans: TransitionSystem[D, A, S], features: (D, S, A) => gnu.trove.map.hash.THashMap[Int, Float],
    lossFactory: LossFunctionFactory[D, A, S], score: Iterable[(D, D)] => List[(String, Double)], utilityFunction: (DAGGEROptions, String, Integer, D, D) => Unit = null) = {
    // Decode all instances, assuming
    val loss = lossFactory.newLossFunction
    val timer = new dagger.util.Timer
    timer.start()

    val (validationLoss, validationScore) = helper(validationData, "val", iter, policy, trans, features, loss, score, utilityFunction)
    val (trainingLoss, trainingScore) = helper(trainingData, "trng", iter, policy, trans, features, loss, score, utilityFunction)

    println(f"Mean Loss (Validation):\t${validationLoss / validationData.size}%.3f")
    println(f"Mean Loss (Training):\t${trainingLoss / trainingData.size}%.3f")
    validationScore foreach (x => println(f"Mean ${x._1} (Validation):\t${x._2}%.3f"))
    trainingScore foreach (x => println(f"Mean ${x._1} (Training):\t${x._2}%.3f"))
    println(s"Time taken for validation:\t$timer")
  }

  def helper(data: Iterable[D], postfix: String, iter: Integer, policy: ProbabilisticClassifierPolicy[D, A, S], trans: TransitionSystem[D, A, S],
    features: (D, S, A) => gnu.trove.map.hash.THashMap[Int, Float], loss: LossFunction[D, A, S], score: Iterable[(D, D)] => List[(String, Double)],
    utilityFunction: (DAGGEROptions, String, Integer, D, D) => Unit = null): (Double, List[(String, Double)]) = {
    val debug = new FileWriter(options.DAGGER_OUTPUT_PATH + "Stats_debug.txt", true)
    val decoded = data.map { d => decode(d, policy, trans, features) }
    val totalLoss = data.zip(decoded).zipWithIndex.map {
      case ((d, decodePair), index) =>
        val (prediction, actions) = decodePair
        prediction match {
          case Some(structure) =>
            if (utilityFunction != null) utilityFunction(options, postfix + "_" + iter, (index + 1), structure, d)
            if (options.DEBUG) {
              debug.write("Target = " + d + "\n")
              debug.write("Prediction = " + structure + "\n")
              debug.write("Actions Taken: \n")
              for (a <- actions) debug.write(a.toString + "\n")
            }
            loss(d, structure, actions)
          case _ => {
            println("No data found for " + d)
            loss.max(d)
          }
        }
    }.foldLeft(0.0)(_ + _)
    val totalScore = score(data.zip(decoded filter { case (prediction, _) => prediction != None } map (_._1.get)))
    debug.close()
    (totalLoss, totalScore)
  }
}

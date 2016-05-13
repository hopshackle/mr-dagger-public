package dagger.core

import java.io.FileWriter

import dagger.ml._
import scala.collection.parallel.{ ParIterable, ForkJoinTaskSupport }
import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.forkjoin.ForkJoinPool
import scala.reflect.ClassTag
import scala.util.Random

/**
 * Created by narad on 6/10/14.
 */
// Dagger[Action, State]
// D = Data
// A = Action
// S = State
class DAGGER[D <: DaggerData[S, A]: ClassTag, A <: TransitionAction[S]: ClassTag, S <: TransitionState: ClassTag](options: DAGGEROptions) {
  val random = new Random(options.RANDOM_SEED)

  def train(data: Iterable[D],
    expert: HeuristicPolicy[D, A, S],
    featureFactory: FeatureFunctionFactory[D, S, A],
    trans: TransitionSystem[D, A, S],
    lossFactory: LossFunctionFactory[D, A, S],
    dev: Iterable[D] = Iterable.empty,
    test: Iterable[D] = Iterable.empty,
    score: Iterable[(D, D)] => List[(String, Double)],
    actionToString: (A => String) = null,
    stringToAction: (String => A) = null,
    utilityFunction: (DAGGEROptions, String, Integer, D, D) => Unit = null,
    startingClassifier: MultiClassClassifier[A] = null): MultiClassClassifier[A] = {

    // Begin DAGGER training
    var instances = new ArrayBuffer[Instance[A]]
    var classifier = startingClassifier
    var policy = new ProbabilisticClassifierPolicy[D, A, S](classifier)
    for (i <- 1 to options.DAGGER_ITERATIONS) {
      val maxTrainingSizeThisIteration = Math.min(options.MAX_TRAINING_SIZE, options.MIN_TRAINING_SIZE + (i - 1) * options.TRAINING_SIZE_INC)
      val filteredData = data filter (_.size <= maxTrainingSizeThisIteration)
      println(s"${filteredData.size} smallest data samples being used out of ${data.size}")
      if (i > 1) options.USE_EXPERT_ON_ROLLOUT_AFTER += options.EXPERT_HORIZON_INCREMENT
      val prob = options.INITIAL_EXPERT_PROB * math.pow(1.0 - options.POLICY_DECAY, i - 1)
      println("DAGGER iteration %d of %d with P(oracle) = %.2f".format(i, options.DAGGER_ITERATIONS, prob))
      val newInstances = collectInstances(filteredData, expert, policy, featureFactory, trans, lossFactory, prob, i, utilityFunction)
      if (options.getBoolean("--fileCache", false) && actionToString != null) {
        writeInstancesToFile(newInstances, i, actionToString)
      }
      classifier = if (options.RETRAIN_EACH_CLASSIFIER) null.asInstanceOf[MultiClassClassifier[A]] else classifier
      if (!options.getBoolean("--fileCache", false) || actionToString == null) {
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
            if (dev.nonEmpty) stats(filteredData, j, dev, test, new ProbabilisticClassifierPolicy[D, A, S](classifier), expert, trans,
              featureFactory.newFeatureFunction, lossFactory, score, utilityFunction)
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

      val classifierToTest = if (options.AVERAGING) classifier.applyAveraging else classifier
      val policyToTest = new ProbabilisticClassifierPolicy[D, A, S](classifierToTest)
      policy = new ProbabilisticClassifierPolicy[D, A, S](classifier)
      // Optionally discard old training instances, as in pure imitation learning
      if (options.DISCARD_OLD_INSTANCES) instances.clear()
      if (dev.nonEmpty && !options.PLOT_LOSS_PER_ITERATION) stats(data, i, dev, test, policyToTest, expert, trans, featureFactory.newFeatureFunction, lossFactory, score, utilityFunction)
      classifierToTest.writeToFile(options.DAGGER_OUTPUT_PATH + "Classifier_" + i + ".txt", actionToString)
    }

    if (options.AVERAGING) classifier.applyAveraging else classifier
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

    val LOLSDet = options.ALGORITHM == "LOLSDet" || options.ALGORITHM == "DILDO"
    val debug = if (options.DEBUG) new FileWriter(options.DAGGER_OUTPUT_PATH + "CollectInstances_debug_" + iteration + "_" + f"$prob%.3f" + ".txt") else null
    var lossOnTestSet = List[Double]()
    var processedSoFar = 0
    val dataWithIndex = data.zipWithIndex
    val original_MAX_ACTIONS = options.MAX_ACTIONS

    val allData = fork(dataWithIndex, options.NUM_CORES).flatMap {
      case (d, dcount) =>
        val MAX_ACTIONS = Math.max(original_MAX_ACTIONS, d.size * options.ACTIONS_PER_SIZE)
        options.MAX_ACTIONS = MAX_ACTIONS
        val instances = new ArrayBuffer[Instance[A]]
        // We create new Loss and Feature functions each time for thread-safety as they cache some results for performance reasons
        val loss = lossFactory.newLossFunction
        val featFn = featureFactory.newFeatureFunction

        // Use policies to fully construct (unroll) instance from start state
        val (_, expertActions, _) = if (options.UNROLL_EXPERT_FOR_LOSS) unroll(d, expert, policy, trans.init(d), trans, featFn, 1.0, debug = debug) else (0, Array[A](), 0)
        val (predEx, predActions, expertUse) = unroll(d, expert, policy, trans.init(d), trans, featFn, prob, true, debug)
        if (options.DEBUG) {
          debug.write("Actions Taken:\n")
          (predActions zip expertUse) foreach (x => debug.write(x._1 + " : " + x._2 + "\n"))
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

        var state = trans.init(d)

        loss.setSamples(options.NUM_SAMPLES * (if (LOLSDet && policy.classifier != null && !options.ORACLE_LOSS) 2 else 1))
        val allInstances = predActions.zipWithIndex map {
          case (a, actionNumber) =>

            // Find all actions permissible for current state
            val allPermissibleActions = trans.permissibleActions(state)
            val nextExpertAction = expert.chooseTransition(d, state)
            val nextPolicyActionsAndScores = if (policy.classifier != null)
              predictUsingPolicy(d, state, policy, allPermissibleActions, featFn.features)
            else {
              // pick a non-expert action at random
              val numberToPick = options.ROLLOUT_LIMIT
              val excludingExpertChoice = allPermissibleActions.filterNot { x => x == nextExpertAction }
              val allChoices = (if (excludingExpertChoice.size > numberToPick) {
                val randomNumbers = Random.shuffle[Int, Seq](Range(0, excludingExpertChoice.size)) take numberToPick
                randomNumbers map excludingExpertChoice toArray
              } else excludingExpertChoice)
              allChoices.toSeq map { x => (x, 0.0f) }
            }

            val nextPolicyActions = nextPolicyActionsAndScores map { x => x._1 }
            val policyActionScores = nextPolicyActionsAndScores.toMap
            val permissibleActions = options.REDUCED_ACTION_SPACE match {
              case true if (nextPolicyActions contains nextExpertAction) => nextPolicyActions.toArray
              case true => (nextPolicyActions ++ Seq(nextExpertAction)).toArray
              case false if (allPermissibleActions contains nextExpertAction) => allPermissibleActions
              case false => allPermissibleActions :+ nextExpertAction
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
                val (_, expertActionsFromHere, _) = if (options.UNROLL_EXPERT_FOR_LOSS) unroll(d, expert, policy, stateCopy, trans, featFn, 1.0) else (0, Array[A](), 0)
                if (options.APPROXIMATE_LOSS) {
                  trans.approximateLoss(datum = d, state = state, action = l)
                } else if (LOLSDet && policy.classifier != null && !options.ORACLE_LOSS) {
                  val (sampledExEx, sampledActionsEx, _) = unroll(d, expert, policy, stateCopy, trans, featFn, 1.0) // uses expert
                  val expertLoss = calculateAndLogLoss(sampledExEx, sampledActionsEx, Array(true), expertActionsFromHere, l, nextExpertAction)
                  val (sampledExLP, sampledActionsLP, _) = unroll(d, expert, policy, stateCopy, trans, featFn, 0.0) // uses learned policy
                  val policyLoss = calculateAndLogLoss(sampledExLP, sampledActionsLP, Array(false), expertActionsFromHere, l, nextExpertAction)
                  expertLoss * prob + policyLoss * (1.0 - prob)
                } else {
                  // Unroll from current state until completion
                  val (sampledEx, sampledActions, expertInSample) = if (!options.ORACLE_LOSS) unroll(d, expert, policy, stateCopy, trans, featFn, prob) else (Some(d), Array[A](), Array[Boolean]())
                  calculateAndLogLoss(sampledEx, sampledActions, expertInSample, expertActionsFromHere, l, nextExpertAction)
                }
              }.foldLeft(0.0)(_ + _) / options.NUM_SAMPLES // average the label loss for all samples
            }
            // Reduce all costs until the min cost is 0

            val coachedCosts = costs
            val min = costs.minBy(_ * 1.0)
            val costsWithoutOffset = costs.map(x => (x - min))
            val normedCosts = if (options.BINARY_LOSS) costsWithoutOffset map (x => if (x == 0.0) 0.0 else 1.0) else costsWithoutOffset

            val allFeatures = permissibleActions map (a => featFn.features(d, state, a))
            val weightLabels = permissibleActions map (_.getMasterLabel.asInstanceOf[A])
            val instance = new Instance[A](allFeatures(0)._1, extractParameterFeatures(allFeatures), permissibleActions, weightLabels, normedCosts map (_.toFloat))

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

            loss.clearCache

            // Progress to next state in the predicted path
            state = a(state)
            instance
        }
        options.MAX_ACTIONS = original_MAX_ACTIONS
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
    featureFunction: FeatureFunction[D, S, A],
    probability: Double = 1.0, rollIn: Boolean = false, debug: FileWriter = null): (Option[D], Array[A], Array[Boolean]) = {

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
      val temp = trans.permissibleActions(state)
      val permissibleActions = if (temp.isEmpty) {
        println("No permissible actions for " + ex)
        println("From state " + state)
        //       return (Some(ex.getEmptyExample.asInstanceOf[D]), actions.toArray, expertUsed.toArray)
        Array(ex.getDefaultAction)
      } else temp
      val policy = (rollIn, options.EXPERT_HORIZON_ROLLOUT, actionsTaken >= options.USE_EXPERT_ON_ROLLOUT_AFTER, classifierPolicy.classifier == null, random.nextDouble <= prob) match {
        case (_, _, _, true, _) =>
          expertUsed += true; expertPolicy
        case (true, _, _, _, true) =>
          expertUsed += true; expertPolicy
        case (true, _, _, _, false) =>
          expertUsed += false; classifierPolicy
        case (false, false, _, _, false) =>
          expertUsed += false; classifierPolicy
        case (false, false, _, _, true) =>
          expertUsed += true; expertPolicy
        case (false, true, false, _, _) =>
          expertUsed += false; classifierPolicy
        case (false, true, true, _, _) =>
          expertUsed += true; expertPolicy
      }
      val expertAction = if (expertPolicy != null) expertPolicy.predict(ex, state) else null.asInstanceOf[A]
      val actionsAndScores = if (classifierPolicy.classifier != null)
        predictUsingPolicy(ex, state, classifierPolicy, permissibleActions, featureFunction.features) sortWith {
          case ((a1: A, s1: Float), (a2: A, s2: Float)) => s1 > s2
        }
      else Seq()

      val a = policy match {
        case x: HeuristicPolicy[D, A, S] =>
          expertAction
        case y: ProbabilisticClassifierPolicy[D, A, S] =>
          actionsAndScores.head._1
      }

      if (options.DEBUG && debug != null) {
        debug.write("State:" + state + "\nExpertAction: " + expertAction + "\n")
        debug.write("\n" + (actionsAndScores map { case (action, score) => f"$action $score%.3f\t" }).mkString(" ") + "\n")
        if (options.ROLLOUT_THRESHOLD > 0.0) {
          if (options.DETAIL_DEBUG && a != expertAction) {
            for (a <- actionsAndScores map { _._1 }) {
              debug.write(a + "\n")
              import scala.collection.JavaConversions._
              val features = featureFunction.features(ex, state, a)
              val scalaMap: Map[Int, Float] = features._1 ++ features._2
              for ((k, v) <- scalaMap) {
                val feature = featureFunction.featureName(k)
                val padding = " " * (50 - feature.size)
                val weight = classifierPolicy.classifier.weightOf(a.getMasterLabel.asInstanceOf[A], k)
                debug.write(f"$feature $padding $v%.2f \t Weight: $weight%+.3f\n")
              }
              debug.write("\n")
            }
          }
        }
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

  def extractParameterFeatures(featFnOutput: Array[(Map[Int, Float], Map[Int, Float])]): Map[Int, Map[Int, Float]] = {
    (featFnOutput.zipWithIndex filterNot (_._1._2.isEmpty) map { case (f, i) => (i -> f._2) }).toMap
  }

  def predictUsingPolicy(ex: D, state: S, policy: ProbabilisticClassifierPolicy[D, A, S], permissibleActions: Array[A],
    featureFunction: (D, S, A) => (Map[Int, Float], Map[Int, Float])): Seq[(A, Float)] = {
    val weightLabels = permissibleActions map (_.getMasterLabel.asInstanceOf[A])
    val features = permissibleActions map (a => featureFunction(ex, state, a))
    val instance = new Instance[A](features(0)._1, extractParameterFeatures(features),
      permissibleActions, weightLabels, permissibleActions.map(_ => 0.0f))
    val prediction = policy.predict(ex, instance, state, options.ROLLOUT_THRESHOLD, options.ROLLOUT_LIMIT)
    prediction
  }

  def trainFromInstances(instances: Iterable[Instance[A]], actions: Array[A], old: MultiClassClassifier[A]): MultiClassClassifier[A] = {
    val weightLabels = actions map (_.getMasterLabel.asInstanceOf[A])
    old match {
      case c: AROWClassifier[A] => AROW.train[A](instances, weightLabels, options, Some(c))
      case _ => AROW.train[A](instances, weightLabels, options)
    }
  }
  def decode(ex: D, classifierPolicy: ProbabilisticClassifierPolicy[D, A, S], expert: HeuristicPolicy[D, A, S],
    trans: TransitionSystem[D, A, S], featFn: FeatureFunction[D, S, A], debug: FileWriter): (Option[D], Array[A]) = {
    unroll(ex, expert, classifierPolicy, start = trans.init(ex), trans, featFn, probability = 0.0, true, debug) match { case (a, b, c) => (a, b) }
  }

  def fork[T](data: Iterable[T], forkSize: Int): ParIterable[T] = {
    System.err.println("Parallelizing to %d cores...".format(forkSize))
    val par = data.par
    par.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(forkSize))
    par
  }

  def stats(trainingData: Iterable[D], iter: Integer, validationData: Iterable[D], testData: Iterable[D], policy: ProbabilisticClassifierPolicy[D, A, S],
    expert: HeuristicPolicy[D, A, S], trans: TransitionSystem[D, A, S], features: FeatureFunction[D, S, A],
    lossFactory: LossFunctionFactory[D, A, S], score: Iterable[(D, D)] => List[(String, Double)], utilityFunction: (DAGGEROptions, String, Integer, D, D) => Unit = null) = {
    // Decode all instances, assuming
    val loss = lossFactory.newLossFunction
    val timer = new dagger.util.Timer
    timer.start()

    println("Starting validation analysis")
    val (validationLoss, validationScore) = helper(validationData, "val", iter, policy, expert, trans, features, loss, score, utilityFunction)
    println(f"Mean Loss (Validation):\t${validationLoss / validationData.size}%.3f")
    validationScore foreach (x => println(f"Mean ${x._1} (Validation):\t${x._2}%.3f"))

    if (options.LOG_TRAINING_STATS) {
      val (trainingLoss, trainingScore) = helper(trainingData, "trng", iter, policy, expert, trans, features, loss, score, utilityFunction)
      println(f"Mean Loss (Training):\t${trainingLoss / trainingData.size}%.3f")
      trainingScore foreach (x => println(f"Mean ${x._1} (Training):\t${x._2}%.3f"))
    }

    helper(testData, "test", iter, policy, null, trans, features, loss, score, utilityFunction)

    println(s"Time taken for validation:\t$timer")
  }

  def helper(data: Iterable[D], postfix: String, iter: Integer, policy: ProbabilisticClassifierPolicy[D, A, S], expert: HeuristicPolicy[D, A, S],
    trans: TransitionSystem[D, A, S], featFn: FeatureFunction[D, S, A], loss: LossFunction[D, A, S], score: Iterable[(D, D)] => List[(String, Double)],
    utilityFunction: (DAGGEROptions, String, Integer, D, D) => Unit = null): (Double, List[(String, Double)]) = {
    val debug = if (options.DEBUG) new FileWriter(options.DAGGER_OUTPUT_PATH + "Validation_debug_" + postfix + "_" + iter + ".txt", true) else null

    val decoded = data.map {
      d => decode(d, policy, expert, trans, featFn, debug)
    }
    val totalLoss = data.zip(decoded).zipWithIndex.map {
      case ((d, decodePair), index) =>
        val (prediction, actions) = decodePair
        prediction match {
          case Some(structure) =>
            if (index % 200 == 0) println("Calculating " + (index + 1))
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
    if (debug != null) debug.close()
    (totalLoss, totalScore)
  }
}

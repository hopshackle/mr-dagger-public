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
class DAGGER[D: ClassTag, A <: TransitionAction[S]: ClassTag, S <: TransitionState: ClassTag](options: DAGGEROptions) {
  val random = new Random(options.RANDOM_SEED)

  def train(data: Iterable[D],
    expert: HeuristicPolicy[D, A, S],
    featureFactory: FeatureFunctionFactory[D, S, A],
    trans: TransitionSystem[D, A, S],
    lossFactory: LossFunctionFactory[D, A, S],
    dev: Iterable[D] = Iterable.empty,
    score: Iterable[(D, D)] => Double,
    utilityFunction: (DAGGEROptions, String, D) => Unit = null): MultiClassClassifier[A] = {
    // Construct new classifier and uniform classifier policy
    //    val dataSize = data.size
    //    val cache = new mutable.HashMap[S, Array[Double]]

    //  def features = (d: D, s: S, a: A) => Map[Int, Double]()
    // Begin DAGGER training
    val instances = new ArrayBuffer[Instance[A]]
    var classifier = null.asInstanceOf[MultiClassClassifier[A]]
    var policy = new ProbabilisticClassifierPolicy[D, A, S](classifier)
    for (i <- 1 to options.DAGGER_ITERATIONS) {
      val prob = math.pow(1.0 - options.POLICY_DECAY, i - 1)
      println("DAGGER iteration %d of %d with P(oracle) = %.2f".format(i, options.DAGGER_ITERATIONS, prob))
      instances ++= collectInstances(data, expert, policy, featureFactory, trans, lossFactory, prob, i, utilityFunction)
      println("DAGGER iteration - training classifier on " + instances.size + " total instances.")
      classifier = trainFromInstances(instances, trans.actions, old = classifier)
      policy = new ProbabilisticClassifierPolicy[D, A, S](classifier)
      // Optionally discard old training instances, as in pure imitation learning
      if (options.DISCARD_OLD_INSTANCES) instances.clear()
      if (dev.nonEmpty) stats(data, dev, policy, trans, featureFactory.newFeatureFunction.features, lossFactory, score, utilityFunction)
    }
    classifier
  }

  def collectInstances(data: Iterable[D], expert: HeuristicPolicy[D, A, S], policy: ProbabilisticClassifierPolicy[D, A, S],
    featureFactory: FeatureFunctionFactory[D, S, A], trans: TransitionSystem[D, A, S], lossFactory: LossFunctionFactory[D, A, S],
    prob: Double = 1.0, iteration: Int, utilityFunction: (DAGGEROptions, String, D) => Unit): Array[Instance[A]] = {
    val timer = new dagger.util.Timer
    timer.start()
    // Compute the probability of choosing the oracle policy in this round
    //      val prob = math.pow(1.0 - options.POLICY_DECAY, i-1)
    //      println("DAGGER iteration %d of %d with P(oracle) = %.2f".format(i, options.DAGGER_ITERATIONS, prob))
    // Keep statistics on # of failed unrolls and the accuracy of predicted structures
    //    var numFailedUnrolls = 0
    //    var numCorrectUnrolls = 0
    val file = if (options.SERIALIZE) new FileWriter(options.DAGGER_OUTPUT_PATH + options.DAGGER_SERIALIZE_FILE) else null
    val debug = new FileWriter(options.DAGGER_OUTPUT_PATH + "CollectInstances_debug_" + f"$prob%.3f" + ".txt")
    var lossOnTestSet = List[Double]()
    var processedSoFar = 0
    val dataWithIndex = data.zipWithIndex
    val allData = fork(dataWithIndex, options.NUM_CORES).flatMap {
      case (d, dcount) =>
        val instances = new ArrayBuffer[Instance[A]]
        // We create new Loss and Feature functions each time for thread-safety as they cache some results for performance reasons
        val loss = lossFactory.newLossFunction
        val featFn = featureFactory.newFeatureFunction
        // Use policies to fully construct (unroll) instance from start state
        val (_, expertActions, _) = unroll(d, expert, policy, trans.init(d), trans, featFn.features, 1.0)
        val (predEx, predActions, expertUse) = unroll(d, expert, policy, trans.init(d), trans, featFn.features, prob)
        if (options.DEBUG) debug.write("Initial State:" + trans.init(d) + "\n")
        if (options.DEBUG) { debug.write("Actions Taken:\n"); (predActions zip expertUse) foreach (x => debug.write(x._1 + " : " + x._2 + "\n")) }

        val totalLoss = predEx match {
          case None => 1.0
          case Some(output) => if (utilityFunction != null) utilityFunction(options, iteration + "_" + (dcount + 1).toString, output); loss(output, d, predActions, expertActions)
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
        loss.setSamples(options.NUM_SAMPLES)
        val allInstances = predActions.map { a =>
          // Find all actions permissible for current state
          val nextExpertAction = expert.chooseTransition(d, state)
          if (options.DEBUG) { debug.write(state + "\n"); debug.flush }
          val permissibleActions = trans.permissibleActions(state)

          // Compute a cost for each permissible action
          val costs = permissibleActions.map { l =>
            (1 to options.NUM_SAMPLES).map { s =>
              // Create a copy of the state and apply the action for the cost calculation
              var stateCopy = state
              stateCopy = l(stateCopy)
              //        if (options.EXPERT_APPROXIMATION) {
              //          loss(gold = d, test = trans.expertApprox(d, stateCopy), testActions = Array(), expertActions)
              //        }
              if (options.APPROXIMATE_LOSS) {
                trans.approximateLoss(datum = d, state = state, action = l)
              } else {
                // Unroll from current state until completion
                val (_, expertActionsFromHere, _) = unroll(d, expert, policy, stateCopy, trans, featFn.features, 1.0)
                val (sampledEx, sampledActions, expertInSample) = unroll(d, expert, policy, stateCopy, trans, featFn.features, prob)
                // If the unrolling is successful, calculate loss with respect to gold structure
                // Otherwise use the max cost
                sampledEx match {
                  case Some(structure) =>
                    val ll = loss(gold = d, test = structure, sampledActions, expertActionsFromHere, l, nextExpertAction)
                    if (options.DEBUG) debug.write(f"Loss on action $l = $ll%.3f\n")
                    ll
                  case None =>
                    if (options.DEBUG) debug.write("Failed unroll, loss = " + loss.max(d) + "\n")
                    loss.max(d)
                }
              }
            }.foldLeft(0.0)(_ + _) / options.NUM_SAMPLES // average the label loss for all samples
            //            }
          }
          // Reduce all costs until the min cost is 0

          val min = costs.minBy(_ * 1.0)
          val normedCosts = costs.map(_ - min)
          if (options.DEBUG) debug.write("Actions = " + permissibleActions.mkString(", ") + "\n")
          if (options.DEBUG) debug.write("Original Costs = " + (costs map (i => f"$i%.3f")).mkString(", ") + "\n")
          if (options.DEBUG) debug.write("Normed Costs = " + (normedCosts map (i => f"$i%.3f")).mkString(", ") + "\n")
          if (options.DEBUG) {
            val minAction = permissibleActions(normedCosts.indexOf(0.0))
            debug.write("Expert action: " + nextExpertAction + ", versus min cost action: " + minAction + "\n")
            //          normedCosts = permissibleActions.map(pa => if (pa == chosen) 0.0 else 1.0)
            debug.write("\n")
            debug.flush()
          }
          // Construct new training instance with sampled losses
          val allFeatures = permissibleActions map (a => featFn.features(d, state, a))
          val weightLabels = permissibleActions map (_.getMasterLabel.asInstanceOf[A])
          val instance = new Instance[A](allFeatures.toList, permissibleActions, weightLabels, normedCosts)
          loss.clearCache
          //        if (options.SERIALIZE) file.write(instance.toSerialString + "\n\n") else instances += instance

          // Progress to next state in the predicted path
          state = a(state)
          instance
        }
        this.synchronized {
          processedSoFar += 1
          if (processedSoFar % options.DAGGER_PRINT_INTERVAL == 0) {
            System.err.print("\r..instance %d in %s, average time per instance = %s".format(dcount + 1, timer.toString, timer.toString(divisor = processedSoFar)))
          }
        }
        allInstances
    }.toArray
    if (options.DEBUG) debug.write(f"Mean Loss on test set:\t ${(lossOnTestSet reduce (_ + _)) / lossOnTestSet.size}%.3f")
    debug.close
    allData
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
    featureFunction: (D, S, A) => Map[Int, Double],
    prob: Double = 1.0): (Option[D], Array[A], Array[Boolean]) = {
    val actions = new ArrayBuffer[A]
    val expertUsed = new ArrayBuffer[Boolean]
    var state = start
    var actionsTaken = 0
    while (!trans.isTerminal(state) && actionsTaken < 300) {
      val permissibleActions = trans.permissibleActions(state)
      if (permissibleActions.isEmpty) {
        return (None, actions.toArray, expertUsed.toArray)
      }
      val policy = if (random.nextDouble() <= prob) { expertUsed += true; expertPolicy } else { expertUsed += false; classifierPolicy }
      val a = policy match {
        case x: HeuristicPolicy[D, A, S] => x.predict(ex, state)
        case y: ProbabilisticClassifierPolicy[D, A, S] => {
          val weightLabels = permissibleActions map (_.getMasterLabel.asInstanceOf[A])
          val instance = new Instance[A]((permissibleActions map (a => featureFunction(ex, state, a))).toList,
            permissibleActions, weightLabels, permissibleActions.map(_ => 0.0))
          val prediction = y.predict(ex, instance, state)
          if (prediction.size > 1) prediction(scala.util.Random.nextInt(prediction.size)) else prediction.head
        }
      }

      actions += a
      actionsTaken += 1
      if (actionsTaken == 300) {
        println(s"Unroll terminated at $actionsTaken actions")
        //       println(ex)
        //      println("Actions Taken: ")
        //       println(actions.slice(0, 50))
      }
      state = a(state)
    }
    (Some(trans.construct(state, ex)), actions.toArray, expertUsed.toArray)
  }

  def trainFromInstances(instances: Iterable[Instance[A]], actions: Array[A], old: MultiClassClassifier[A]): MultiClassClassifier[A] = {
    val weightLabels = actions map (_.getMasterLabel.asInstanceOf[A])
    options.CLASSIFIER match {
      case "AROW" => {
        old match {
          case c: AROWClassifier[A] => AROW.train[A](instances, actions, weightLabels, options, Some(c))
          case _ => AROW.train[A](instances, actions, weightLabels, options)
        }
      }
      case "PASSIVE_AGGRESSIVE" => ??? //PassiveAggressive.train[A](instances, actions, options.RATE, random, options)
      case "PERCEPTRON" => ??? // Perceptron.train[A](instances, actions, options.RATE, random, options)
    }
  }
  def decode(ex: D, classifierPolicy: ProbabilisticClassifierPolicy[D, A, S],
    trans: TransitionSystem[D, A, S], featureFunction: (D, S, A) => Map[Int, Double]): (Option[D], Array[A]) = {
    unroll(ex, expertPolicy = null, classifierPolicy, start = trans.init(ex), trans, featureFunction, prob = 0.0) match { case (a, b, c) => (a, b) }
  }

  def fork[T](data: Iterable[T], forkSize: Int): ParIterable[T] = {
    System.err.println("Parallelizing to %d cores...".format(forkSize))
    val par = data.par
    par.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(forkSize))
    par
  }

  def stats(trainingData: Iterable[D], validationData: Iterable[D], policy: ProbabilisticClassifierPolicy[D, A, S], trans: TransitionSystem[D, A, S], features: (D, S, A) => Map[Int, Double],
    lossFactory: LossFunctionFactory[D, A, S], score: Iterable[(D, D)] => Double, utilityFunction: (DAGGEROptions, String, D) => Unit = null) = {
    // Decode all instances, assuming
    val loss = lossFactory.newLossFunction
    val timer = new dagger.util.Timer
    timer.start()

    val (validationLoss, validationScore) = helper(validationData, policy, trans, features, loss, score, utilityFunction)
    val (trainingLoss, trainingScore) = helper(trainingData, policy, trans, features, loss, score)

    println(f"Mean Loss (Validation):\t${validationLoss / validationData.size}%.2f")
    println(f"Mean Loss (Training):\t${trainingLoss / trainingData.size}%.2f")
    println(f"Mean Score (Validation):\t${validationScore}%.2f")
    println(f"Mean Score (Training):\t${trainingScore}%.2f")
    println(s"Time taken for validation:\t$timer")
  }

  def helper(data: Iterable[D], policy: ProbabilisticClassifierPolicy[D, A, S], trans: TransitionSystem[D, A, S],
    features: (D, S, A) => Map[Int, Double], loss: LossFunction[D, A, S], score: Iterable[(D, D)] => Double,
    utilityFunction: (DAGGEROptions, String, D) => Unit = null): (Double, Double) = {
    val debug = new FileWriter(options.DAGGER_OUTPUT_PATH + "Stats_debug.txt", true)
    val decoded = data.map { d => decode(d, policy, trans, features) }
    val totalLoss = data.zip(decoded).zipWithIndex.map {
      case ((d, decodePair), index) => //case (d, (prediction, actions)) =>
        val (prediction, actions) = decodePair
        prediction match {
          case Some(structure) =>
            if (utilityFunction != null) utilityFunction(options, "val_" + (index + 1), structure)
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
    val totalScore = score(data.zip(decoded filter{case (prediction, _) => prediction != None} map (_._1.get)))
    debug.close()
    (totalLoss, totalScore)
  }
}

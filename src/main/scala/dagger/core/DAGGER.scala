package dagger.core

import java.io.FileWriter

import dagger.ml._
import scala.collection.parallel.{ParIterable, ForkJoinTaskSupport}
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
class DAGGER[D: ClassTag, A <: TransitionAction[S] : ClassTag, S <: TransitionState : ClassTag](options: DAGGEROptions) {
  val random = new Random(options.RANDOM_SEED)

  def train(data: Iterable[D],
            expert: HeuristicPolicy[D, A, S],
            features: (D, S) => Map[Int, Double],
            trans: TransitionSystem[D, A, S],
            loss: LossFunction[D, A, S], //(D, D) => Double,
            dev: Iterable[D] = Iterable.empty,
            score: Iterable[(D,D)] => Double): MultiClassClassifier[A] = {
    // Construct new classifier and uniform classifier policy
    //    val dataSize = data.size
    //    val cache = new mutable.HashMap[S, Array[Double]]

    // Begin DAGGER training
    val instances = new ArrayBuffer[Instance[A]]
    var classifier = null.asInstanceOf[MultiClassClassifier[A]]
    var policy = new ProbabilisticClassifierPolicy[D, A, S](classifier)
    for (i <- 1 to options.DAGGER_ITERATIONS) {
      val prob = math.pow(1.0 - options.POLICY_DECAY, i-1)
      println("DAGGER iteration %d of %d with P(oracle) = %.2f".format(i, options.DAGGER_ITERATIONS, prob))
      instances ++= collectInstances(data, expert, policy, features, trans, loss) //collectInstances(data, expert, policy, features, trans, loss, prob)
      classifier = trainFromInstances(instances, trans.actions, old = classifier)
      policy = new ProbabilisticClassifierPolicy[D, A, S](classifier)
      // Optionally discard old training instances, as in pure imitation learning
      if (options.DISCARD_OLD_INSTANCES) instances.clear()
      if (dev.nonEmpty) stats(dev, policy, trans, features, loss, score)
    }
    classifier
  }

  def collectInstances(data: Iterable[D], expert: HeuristicPolicy[D, A, S], policy: ProbabilisticClassifierPolicy[D, A, S],
                features: (D, S) => Map[Int, Double], trans: TransitionSystem[D, A, S], loss: LossFunction[D, A, S], prob: Double = 1.0): Array[Instance[A]] = {
    val timer = new dagger.util.Timer
    timer.start()
    // Compute the probability of choosing the oracle policy in this round
    //      val prob = math.pow(1.0 - options.POLICY_DECAY, i-1)
    //      println("DAGGER iteration %d of %d with P(oracle) = %.2f".format(i, options.DAGGER_ITERATIONS, prob))
    // Keep statistics on # of failed unrolls and the accuracy of predicted structures
//    var numFailedUnrolls = 0
//    var numCorrectUnrolls = 0
    val file = if (options.SERIALIZE) new FileWriter(options.DAGGER_OUTPUT_PATH + options.DAGGER_SERIALIZE_FILE) else null
    var dcount = 0
    fork(data, options.NUM_CORES).flatMap { d =>
      dcount += 1
      //for ((d,dcount) <- data.view.zipWithIndex) {
      if (dcount % options.DAGGER_PRINT_INTERVAL == 0) {
        System.err.print("\r..instance %d in %s, average time per instance = %s".format(dcount, timer.toString, timer.toString(divisor = dcount)))
      }
      val instances = new ArrayBuffer[Instance[A]]
      // Use policies to fully construct (unroll) instance from start state
      val (predEx, predActions) = unroll(d, expert, policy, trans.init(d), trans, features, prob)
      // Check that the oracle policy is correct
      //        if (i == 1 && options.CHECK_ORACLE) assert(predEx.get == d, "Oracle failed to produce gold structure...Gold:\n%s\nPredicted:\n%s".format(d, predEx.get))
      //        if (predEx.get == d) numCorrectUnrolls += 1
      // Initialize state from data example
      var state = trans.init(d)
      // For all actions used to predict the unrolled structure...
      //for (a <- predActions)
      predActions.map { a =>
        // Find all actions permissible for current state
        val permissibleActions = trans.permissibleActions(state)
        // If using caching, check for a stored set of costs for this state
        //          val costs: Array[Double] = if (options.CACHING && cache.contains(state)) {
        //            cache(state)
        //          }
        // Compute a cost for each permissible action
        //          else {
        val costs = permissibleActions.map { l =>
          (1 to options.NUM_SAMPLES).map { s =>
            // Create a copy of the state and apply the action for the cost calculation
            var stateCopy = state
            stateCopy = l(stateCopy)
            if (options.EXPERT_APPROXIMATION) {
              loss(gold = d, test = trans.expertApprox(d, stateCopy), testActions = Array())
            }
            if (options.APPROXIMATE_LOSS) {
              trans.approximateLoss(datum = d, state = state, action = l)
            }
            else {
              // Unroll from current state until completion
              val (sampledEx, sampledActions) = unroll(d, expert, policy, stateCopy, trans, features, prob = prob)
              // If the unrolling is successful, calculate loss with respect to gold structure
              // Otherwise use the max cost
              sampledEx match {
                case Some(structure) => {
                  loss(gold = d, test = structure, sampledActions)
                }
                case None => loss.max(d)
              }
            }
          }.foldLeft(0.0)(_ + _) // Sum the label loss for all samples
          //            }
        }
        // Reduce all costs until the min cost is 0
        val min = costs.minBy(_ * 1.0)
        val normedCosts = costs.map(_ - min)

        // Construct new training instance with sampled losses
        val instance = new Instance[A](features(d, state), permissibleActions, normedCosts)

//        if (options.SERIALIZE) file.write(instance.toSerialString + "\n\n") else instances += instance

        // Progress to next state in the predicted path
        state = a(state)
        instance
      }
    }.toArray
   // instances.toArray
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
             featureFunction: (D, S) => Map[Int, Double],
             prob: Double = 1.0): (Option[D], Array[A]) = {
    val actions = new ArrayBuffer[A]
    var state = start
    while (!trans.isTerminal(state)) {
      val permissibleActions = trans.permissibleActions(state)
      //      assert(!permissibleActions.isEmpty, "There are no permissible actions (of %s) for state:\n%s".format(trans.actions.mkString(", "), state))
      if (permissibleActions.isEmpty) {
        return (None, actions.toArray)
      }
      val policy = if (random.nextDouble() <= prob) expertPolicy else classifierPolicy
      val a = policy match {
        case x: HeuristicPolicy[D, A, S] => x.predict(ex, state)
        case y: ProbabilisticClassifierPolicy[D, A, S] => {
          val instance = new Instance[A](featureFunction(ex, state), permissibleActions, permissibleActions.map(_ => 0.0))
          val prediction = y.predict(ex, instance, state)
          if (prediction.size > 1) prediction(scala.util.Random.nextInt(prediction.size)) else prediction.head
        }
      }
      actions += a
      state = a(state)
    }
    (Some(trans.construct(state, ex)), actions.toArray)
  }

  def trainFromInstances(instances: Iterable[Instance[A]], actions: Array[A], old: MultiClassClassifier[A]): MultiClassClassifier[A] = options.CLASSIFIER match {
    case "AROW" => {
      old match {
        case c: AROWClassifier[A] => AROW.train[A](instances, actions, options, Some(c))
        case _ => AROW.train[A](instances, actions, options)
      }
    }
    case "PASSIVE_AGGRESSIVE" => ??? //PassiveAggressive.train[A](instances, actions, options.RATE, random, options)
    case "PERCEPTRON" => ??? // Perceptron.train[A](instances, actions, options.RATE, random, options)
  }

  def decode(ex: D, classifierPolicy: ProbabilisticClassifierPolicy[D, A, S],
             trans: TransitionSystem[D, A, S], featureFunction: (D, S) => Map[Int, Double]): (Option[D], Array[A])  = {
    unroll(ex, expertPolicy = null, classifierPolicy, start = trans.init(ex), trans, featureFunction, prob = 0.0)
  }

  def fork[T](data: Iterable[T], forkSize: Int): ParIterable[T] = {
    System.err.println("Parallelizing to %d cores...".format(forkSize))
    val par = data.par
    par.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(forkSize))
    par
  }

  def stats(data: Iterable[D], policy: ProbabilisticClassifierPolicy[D, A, S], trans: TransitionSystem[D, A, S], features:  (D, S) => Map[Int, Double],
            loss: LossFunction[D, A, S], score: Iterable[(D, D)] => Double) = {
    // Decode all instances, assuming
    val decoded = data.map { d => decode(d, policy, trans, features) }
    val totalLoss = data.zip(decoded).map { case(d, decodePair) => //case (d, (prediction, actions)) =>
      val (prediction, actions) = decodePair
      prediction match {
        case Some(structure) => loss(d, structure, actions)
        case _=> loss.max(d)
      }
    }.foldLeft(0.0)(_+_)
    val totalScore = score(data.zip(decoded.map(_._1.get)))
  }
}


















//    println("DAGGER iteration finished in %s.".format(timer.toString))


  //  def train(data: Iterable[D],
  //            expert: HeuristicPolicy[D, A, S],
  //            features: (D, S) => Map[Int, Double],
  //            trans: TransitionSystem[D, A, S],
  //            loss: LossFunction[D, A, S], //(D, D) => Double,
  //            dev: Iterable[D] = Iterable.empty,
  //            score: (D,D) => Double): MultiClassClassifier[A] = {
  //    // Construct new classifier and uniform classifier policy
  //    var classifier = null.asInstanceOf[MultiClassClassifier[A]]
  //    var policy = new ProbabilisticClassifierPolicy[D,A,S](classifier)
  //    val dataSize = data.size
  //    val cache = new mutable.HashMap[S, Array[Double]]
  //
  //    // Begin DAGGER training
  //    val instances = new ArrayBuffer[Instance[A]]
  //    val timer = new dagger.util.Timer
  //
  //    for (i <- 1 to options.DAGGER_ITERATIONS) {
  //      timer.start()
  //      // Optionally discard old training instances, as in pure imitation learning
  //      if (options.DISCARD_OLD_INSTANCES) instances.clear()
  //      // Compute the probability of choosing the oracle policy in this round
  //      val prob = math.pow(1.0 - options.POLICY_DECAY, i-1)
  //      println("DAGGER iteration %d of %d with P(oracle) = %.2f".format(i, options.DAGGER_ITERATIONS, prob))
  //      // Keep statistics on # of failed unrolls and the accuracy of predicted structures
  //      var numFailedUnrolls = 0
  //      var numCorrectUnrolls = 0
  //      val file = if (options.SERIALIZE) new FileWriter(options.DAGGER_OUTPUT_PATH + options.DAGGER_SERIALIZE_FILE) else null
  //      var dcount = 0
  //      val iter = if (options.PARALLELIZE) {
  //        val par = data.par
  //        par.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(options.NUM_CORES))
  //        par
  //      }
  //      else {
  //        data
  //      }
  //      iter.foreach { d =>
  //        dcount += 1
  //      //for ((d,dcount) <- data.view.zipWithIndex) {
  //        if (dcount % options.DAGGER_PRINT_INTERVAL == 0) {
  //          System.err.print("\r..instance %d, average time per instance = %s".format(dcount, timer.toString(divisor = dcount)))
  //        }
  //        // Use policies to fully construct (unroll) instance from start state
  //        val (predEx, predActions) = unroll(d, expert, policy, trans.init(d), trans, features, prob = prob)
  //        // Check that the oracle policy is correct
  //        if (i == 1 && options.CHECK_ORACLE) assert(predEx.get == d, "Oracle failed to produce gold structure...Gold:\n%s\nPredicted:\n%s".format(d, predEx.get))
  //        if (predEx.get == d) numCorrectUnrolls += 1
  //        // Initialize state from data example
  //        var state = trans.init(d)
  //        // For all actions used to predict the unrolled structure...
  //        for (a <- predActions) {
  //          // Find all actions permissible for current state
  //          val permissibleActions = trans.permissibleActions(state)
  //            // If using caching, check for a stored set of costs for this state
  //            val costs: Array[Double] = if (options.CACHING && cache.contains(state)) {
  //              cache(state)
  //            }
  //            // Compute a cost for each permissible action
  //            else {
  //              permissibleActions.map { l =>
  //                (1 to options.NUM_SAMPLES).map { s =>
  //                  // Create a copy of the state and apply the action for the cost calculation
  //                  var stateCopy = state
  //                  stateCopy = l(stateCopy)
  //                  if (options.EXPERT_APPROXIMATION) {
  //                    loss(gold = d, test = trans.expertApprox(d, stateCopy), testActions = Array())
  //                  }
  //                  if (options.APPROXIMATE_LOSS) {
  //                    trans.approximateLoss(datum = d, state = state, action = l)
  //                  }
  //                  else {
  //                    // Unroll from current state until completion
  //                    val (sampledEx, sampledActions) = unroll(d, expert, policy, stateCopy, trans, features, prob = prob)
  //                    // If the unrolling is successful, calculate loss with respect to gold structure
  //                    // Otherwise use the max cost
  //                    sampledEx match {
  //                      case Some(structure) => {
  //                        loss(gold = d, test = structure, sampledActions)
  //                      }
  //                      case None => loss.max(d)
  //                    }
  //                  }
  //                }.foldLeft(0.0)(_ + _) // Sum the label loss for all samples
  //              }
  //            }
  //            // Reduce all costs until the min cost is 0
  //            val min = costs.minBy(_ * 1.0)
  //            val ncosts = costs.map(_ - min)
  //            // If caching, store costs for this state
  //            if (options.CACHING) {
  //              cache(state) = ncosts
  //            }
  //            // Construct new training instance with sampled losses
  //            val instance = new Instance[A](features(d, state), permissibleActions, ncosts)
  //
  //          if (options.SERIALIZE) file.write(instance.toSerialString + "\n\n") else instances += instance
  //
  //          // Progress to next state in the predicted path
  //          state = a(state)
  //        }
  //      }
  //      println("Collected " + instances.size + "training instances at iteration " + i)
  //      println("Unrolling failed for %d of %d".format(numFailedUnrolls, dataSize))
  //      println("Unrolling was correct for %d of %d".format(numCorrectUnrolls, dataSize))
  //      // Train classifier based on extracted instances (classifier maintains state between rounds)
  //      val trainData = instances
  ////        if (options.SERIALIZE) {
  ////        // Instantiate the lazy loader
  ////        new ChunkReader(options.DAGGER_OUTPUT_PATH + options.DAGGER_SERIALIZE_FILE).view.map(Instance.fromSerialString[A](_))
  ////      }
  ////      else {
  //        // Train from instances in memory
  ////        instances
  ////      }
  //
  //      classifier = options.CLASSIFIER match {
  //        case "AROW" => AROW.train[A](trainData, trans.actions, options = options)
  //        case "PASSIVE_AGGRESSIVE" => PassiveAggressive.train[A](trainData, trans.actions, rate = 0.1, random, options)
  //        case "PERCEPTRON" => Perceptron.train[A](trainData, trans.actions, rate = 0.1, random, options)
  //      }
  //      classifier.writeToFile(options.DAGGER_OUTPUT_PATH + options.MODEL_FILE + "." + i + ".pv")
  //
  //      // Construct a new probabilistic policy based on updated classifier weights
  //      policy = new ProbabilisticClassifierPolicy[D,A,S](classifier)
  //      // Check performance on development data
  //      if (dev.nonEmpty) {
  //        val devLoss = dev.map{ d =>
  //          val (decoded, dactions) = unroll(d, expert, policy, trans.init(d), trans, features, prob=0.0)
  //          decoded match {
  //            case Some(structure) => loss(d, structure, dactions)
  //            case _=> loss.max(d)
  //          }
  //        }.foldLeft(0.0)(_+_)  // Sum losses
  //        println("Average development loss = %.2f".format(devLoss / dev.size))
  //      }
  //      println("DAGGER iteration finished in %s.".format(timer.toString))
  //    }
  //    classifier
  //  }













/*



  def train(data: Iterable[D], expert: HeuristicPolicy[D, A, S], features: (D, S) => Map[Int, Double],
            trans: TransitionSystem[D, A, S], loss: LossFunction[D, A, S], dev: Iterable[D]): MultiClassClassifier[A] = {

    // Construct new classifier and uniform classifier policy
    var classifier = null.asInstanceOf[MultiClassClassifier[A]]
    var policy = new ProbabilisticClassifierPolicy[D, A, S](classifier)
    val dataSize = data.size
    val cache = new mutable.HashMap[S, Array[Double]]

    // Begin DAGGER training
    val instances = new ArrayBuffer[Instance[A]]

    for (i <- 1 to options.DAGGER_ITERATIONS) {
      // Optionally discard old training instances, as in pure imitation learning
      if (options.DISCARD_OLD_INSTANCES) instances.clear()
      // Compute the probability of choosing the oracle policy in this round
      val prob = math.pow(1.0 - options.POLICY_DECAY, i-1)
      println("DAGGER iteration %d of %d with P(oracle) = %.2f".format(i, options.DAGGER_ITERATIONS, prob))
      val tinstances = collectInstances()
      val classifier = trainClassifier(instances)


    }
  }


  def collectInstances(data: Iterable[D]): Array[Instance] = {
    var dcount = 0
    data.view.foreach { d =>
      dcount += 1
      //for ((d,dcount) <- data.view.zipWithIndex) {
      if (dcount % options.DAGGER_PRINT_INTERVAL == 0) {
        System.err.print("\r..instance %d, average time per instance = %s".format(dcount, timer.toString(divisor = dcount)))
      }
      // Use policies to fully construct (unroll) instance from start state
      val (predEx, predActions) = unroll(d, expert, policy, trans.init(d), trans, features, prob = prob)
      // Check that the oracle policy is correct
      if (i == 1 && options.CHECK_ORACLE) assert(predEx.get == d, "Oracle failed to produce gold structure...Gold:\n%s\nPredicted:\n%s".format(d, predEx.get))
//      if (predEx.get == d) numCorrectUnrolls += 1
      // Initialize state from data example
      var state = trans.init(d)
      // For all actions used to predict the unrolled structure...
      for (a <- predActions) {
        // Find all actions permissible for current state
        val permissibleActions = trans.permissibleActions(state)
        // If using caching, check for a stored set of costs for this state
        val costs: Array[Double] = if (options.CACHING && cache.contains(state)) {
          cache(state)
        }
        // Compute a cost for each permissible action
        else {
          permissibleActions.map { l =>
            (1 to options.NUM_SAMPLES).map { s =>
              // Create a copy of the state and apply the action for the cost calculation
              var stateCopy = state
              stateCopy = l(stateCopy)
              if (options.EXPERT_APPROXIMATION) {
                loss(gold = d, test = trans.expertApprox(d, stateCopy), testActions = Array())
              }
              if (options.APPROXIMATE_LOSS) {
                trans.approximateLoss(datum = d, state = state, action = l)
              }
              else {
                // Unroll from current state until completion
                val (sampledEx, sampledActions) = unroll(d, expert, policy, stateCopy, trans, features, prob = prob)
                // If the unrolling is successful, calculate loss with respect to gold structure
                // Otherwise use the max cost
                sampledEx match {
                  case Some(structure) => {
                    loss(gold = d, test = structure, sampledActions)
                  }
                  case None => loss.max(d)
                }
              }
            }.foldLeft(0.0)(_ + _) // Sum the label loss for all samples
          }
        }
        // Reduce all costs until the min cost is 0
        val min = costs.minBy(_ * 1.0)
        val ncosts = costs.map(_ - min)
        // If caching, store costs for this state
        if (options.CACHING) {
          cache(state) = ncosts
        }
        // Construct new training instance with sampled losses
        val instance = new Instance[A](features(d, state), permissibleActions, ncosts)

        if (options.SERIALIZE) file.write(instance.toSerialString + "\n\n") else instances += instance

        // Progress to next state in the predicted path
        state = a(state)
      }
    }
  }




    if (options.TRAIN_FROM_ORACLE) {
      println("Training directly from Oracle decisions.")
      classifier = AROW.train[A](oracleInstances(data, trans, features), trans.actions, random, options)
      return classifier
    }


    if (options.ACTIVE_LEARNING && i > 1 && meetsActiveCriteria(d, state, trans, features, permissibleActions, classifier, options.ACTIVE_THRESHOLD)) {
            new Instance[A](features(d, state), permissibleActions, permissibleActions.map(pa => if (pa == a) 0.0 else 1.0))
          }
          else {




  def meetsActiveCriteria(d: D, state: S, trans: TransitionSystem[D,A,S], features: (D, S) => Map[Int, Double], permissible: Array[A], classifier: AROWClassifier[A], threshold: Double): Boolean = {
    val a = trans.chooseTransition(d, state)
//    println(permissible.mkString(", "))
//    println(state)
//    println(d)
    val i = new Instance[A](features(d, state), permissible)
//    println(i.feats.mkString("\n"))
    val prediction = classifier.predict(i)
    prediction.maxLabels.size == 1 && prediction.maxLabels.head == a && prediction.maxScore > threshold
  }




  def unrollWithEarlyStopping(ex: D,
             expertPolicy: HeuristicPolicy[D, A, S],
             classifierPolicy: ProbabilisticClassifierPolicy[D, A, S],
             start: S, trans: TransitionSystem[D, A, S],
             featureFunction: (D, S) => Map[Int, Double],
             prob: Double = 1.0): (Option[D], Array[A]) = {
    val actions = new ArrayBuffer[A]
    var state = start
    while (!trans.isTerminal(state)) {
      val permissibleActions = trans.permissibleActions(state)
      //      assert(!permissibleActions.isEmpty, "There are no permissible actions (of %s) for state:\n%s".format(trans.actions.mkString(", "), state))
      if (permissibleActions.isEmpty) {
        return (None, actions.toArray)
      }
      val policy = if (random.nextDouble() <= prob) expertPolicy else classifierPolicy
      val a = policy match {
        case x: HeuristicPolicy[D, A, S] => {
          x.predict(ex, state)
        }
        case y: ProbabilisticClassifierPolicy[D, A, S] => {
          val instance = new Instance[A](featureFunction(ex, state), permissibleActions, permissibleActions.map(_ => 0.0))
          val prediction = y.predict(ex, instance, state)
          if (prediction.size > 1) prediction(scala.util.Random.nextInt(prediction.size)) else prediction.head
        }
      }
      actions += a
      state = a(state)
    }
    (Some(trans.construct(state, ex)), actions.toArray)
  }



trait EarlyStopping[S] {

   def shouldTerminate(s: S): Boolean
}


 */












//            decay: Double = 0.3,
//            daggerIters: Int=10,
//            arowIters: Int=55,
//            numSamples: Int = 5,
//            rate: Int= -1): AROW[A] = {
//    val maxCost = 1.0





//    while (!state.isTerminal) {
//      println(state)
//      println(actions.size)
//      println


//            init: (D) => S,
//            construct: (S) => D,
//            allActions: Array[A],

//              println("min = " + min)
//              println("costs = [%s]".format(costs.mkString(", ")))
//              println("norm costs = [%s]".format(ncosts.mkString(", ")))
//              println("Completed step %d of %d".format())


//class TransitionSystem[D: ClassTag, A <: Action[TransitionState] : ClassTag, S <: TransitionState : ClassTag] {

//class Dagger[X:ClassTag, Array[X]: ClassTag]

//class DAGGER[D: ClassTag, A <: Action: ClassTag, S: ClassTag] {


//               if (l.isPermissible(state)) {



//                 println("Gold: " + ex)
//                 println("Sampled: " + sampledEx)
//                 println("loss = " + loss(ex, sampledEx))
//

//               }
//               else {
//                 1.0  // Would not account for losses that vary sentence to sentence
//               }



//     val goldActions = predictAll(ex, expert, policy, state, allActions, featureFunction, prob=1.0)
//     println("Accuracy of predicted actions: %f".format(d, ))

//         while (!state.isTerminal) {
//           for (s <- 1 to numSamples) {
//             val instance = new Instance[A](featureFunction(ex, state), allActions)
//             val action = if (math.random <= math.pow(1 - decay, i-1)) {
//               expert.predict(ex, state)
//             } else {
//               val prediction = policy.predict(ex, instance, state)
//               if (prediction.size > 1) prediction(scala.util.Random.nextInt(prediction.size)) else prediction.head
//             }
//             val losses = allActions.map(loss(_,action))
//             instances += new Instance[A](featureFunction(ex, state), allActions, losses)
//             state = action(state)
//           }
//         }


//  def createInstance(state: TransitionState): Instance[T] = {
//    new Instance(featureFunction(ex, state), allActions, alLActions.map())
//  }


/*

, history: Array[Action[T]] = Array())

           val action = policy.predict(d)
         val actions = policy.predict(d)
         actions.zipWith{ case(action, t) =>
           val feats = featureFunction(d, action)
           for (a <- actions) {
             val future = policy.predict(d, actionts.slice(0,t+1))

           }
         }
 */
package dagger.core

import java.io.FileWriter

import dagger.ml._

import scala.collection.parallel.{ ForkJoinTaskSupport, ParIterable }
import scala.concurrent.forkjoin.ForkJoinPool

// import ml.wolfe.nlp.io.ChunkReader

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Created by narad on 4/6/15.
 */
class OracleExtractor[D: ClassTag, A <: TransitionAction[S]: ClassTag, S <: TransitionState: ClassTag](options: DAGGEROptions) {

  val helperDagger = new DAGGER[D, A, S](options)

  def instances(data: Iterable[D],
    trans: TransitionSystem[D, A, S],
    featureFactory: FeatureFunctionFactory[D, S, A],
    printInterval: Int = 1000): Iterable[Instance[A]] = {

    val timer = new dagger.util.Timer
    timer.start()
    val instances = helperDagger.fork(data, options.NUM_CORES).flatMap { d =>
      val featFn = featureFactory.newFeatureFunction
      var s = trans.init(d)
      val tinstances = new ArrayBuffer[Instance[A]]
      while (!trans.isTerminal(s)) {
        val permissibleActions = trans.permissibleActions(s)
        assert(permissibleActions.nonEmpty, "No permissible actions found for state:\n%s".format(s))
        val a = trans.chooseTransition(d, s)
        assert(permissibleActions.contains(a), "Oracle chose action (%s) not considered permissible by transition system for state:\n%s.".format(a, s))
        val costs = permissibleActions.map(pa => if (pa == a) 0.0f else 1.0f)
        val weightLabels = permissibleActions map (_.getMasterLabel.asInstanceOf[A])
        tinstances += new Instance[A]((permissibleActions map (a => featFn.features(d, s, a))).toList, permissibleActions, weightLabels, costs)
        s = a(s)
      }
      tinstances
    }
    timer.stop()
    println("Extracted oracle instances in %s.".format(timer.toString))
    val arrayInstance = instances.toArray
    println("Converted to Array")
    arrayInstance.toIterable
  }

  def train(data: Iterable[D],
    expert: HeuristicPolicy[D, A, S],
    featureFactory: FeatureFunctionFactory[D, S, A],
    lossFactory: LossFunctionFactory[D, A, S],
    trans: TransitionSystem[D, A, S],
    dev: Iterable[D] = Iterable.empty,
    score: Iterable[(D, D)] => Double,
    options: DAGGEROptions,
    utilityFunction: (DAGGEROptions, Int, Int, D) => Unit = null): MultiClassClassifier[A] = {

    val inst = instances(data, trans, featureFactory)
    println("Oracle Extractor - training classifier on " + inst.size + " total instances.")
    val classifier = helperDagger.trainFromInstances(inst, trans.actions, null)
    val policy = new ProbabilisticClassifierPolicy[D, A, S](classifier)

    if (dev.nonEmpty) helperDagger.stats(data, dev, policy, trans, featureFactory.newFeatureFunction.features, lossFactory, score)

    classifier
  }

}

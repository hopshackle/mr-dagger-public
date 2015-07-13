package dagger.core

import scala.reflect.ClassTag

/**
 * Created by narad on 10/18/14.
 */
abstract class LossFunction[D: ClassTag, A <: TransitionAction[S] : ClassTag, S <: TransitionState : ClassTag]  {
  
  def clearCache: Unit = {} 
  // default implementation is to do nothing. Only needs to be overridden if the implementing Loss function
  // actually supports caching
  def setSamples(samples: Int): Unit = {}

  def apply(gold: D, test: D, testActions: Array[A], expertActions: Array[A], trialAction: A, lastExpertAction: A): Double = this.apply(gold, test, testActions, expertActions)
  // This is the one we now call in DAGGER - but the final two arguments are only used for AMR 
  // So for backwards compatibility, we provide a default implementation that calls the original signature
  
  def apply(gold: D, test: D, testActions: Array[A]): Double
  
  def apply(gold: D, test: D, testActions: Array[A], expertActions: Array[A]): Double = this.apply(gold, test, testActions)

  def max(gold: D): Double
}

abstract class LossFunctionFactory[D: ClassTag, A <: TransitionAction[S] : ClassTag, S <: TransitionState : ClassTag] {
  def newLossFunction: LossFunction[D, A, S]
}
package dagger.core
import scala.collection.Map

import scala.reflect.ClassTag
abstract class FeatureFunctionFactory[D: ClassTag, S <: TransitionState: ClassTag, A <: TransitionAction[S]: ClassTag] {
  def newFeatureFunction: FeatureFunction[D, S, A]
}

abstract class FeatureFunction[D: ClassTag, S <: TransitionState: ClassTag, A <: TransitionAction[S]: ClassTag] {
  def features(data: D, state: S, action: A): Map[Int, Float]
}
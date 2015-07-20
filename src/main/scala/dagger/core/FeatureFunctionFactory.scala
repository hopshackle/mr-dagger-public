package dagger.core
import gnu.trove.map.hash.THashMap

import scala.reflect.ClassTag
abstract class FeatureFunctionFactory[D: ClassTag, S <: TransitionState: ClassTag, A <: TransitionAction[S]: ClassTag] {
  def newFeatureFunction: FeatureFunction[D, S, A]
}

abstract class FeatureFunction[D: ClassTag, S <: TransitionState: ClassTag, A <: TransitionAction[S]: ClassTag] {
  def features(data: D, state: S, action: A): gnu.trove.map.hash.THashMap[Int, Float]
}
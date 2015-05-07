package coref.dagger

import scala.reflect.ClassTag

/**
 * Created by narad on 10/18/14.
 */
abstract class LossFunction[D: ClassTag, A <: TransitionAction[S] : ClassTag, S <: TransitionState : ClassTag]  {

  def apply(gold: D, test: D, testActions: Array[A]): Double

  def max(gold: D): Double
}
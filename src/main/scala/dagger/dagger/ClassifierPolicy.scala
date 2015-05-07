package coref.dagger

import coref.ml.{Instance, MultiClassClassifier}


/**
 * Created by narad on 6/19/14.
 */

abstract class Policy[D, A <: TransitionAction[S], S <: TransitionState] {}

abstract class HeuristicPolicy[D, A <: TransitionAction[S], S <: TransitionState] extends Policy[D, A, S] {

  def predict(ex: D, state: S) = chooseTransition(ex, state)

  def chooseTransition(instance: D, state: S): A

}

case class ProbabilisticClassifierPolicy[D, A <: TransitionAction[S], S <: TransitionState](classifier: MultiClassClassifier[A]) extends Policy[D, A, S] {

  def predict(ex: D, instance: Instance[A], state: S): Seq[A] = {
      classifier.predict(instance).maxLabels
  }
}
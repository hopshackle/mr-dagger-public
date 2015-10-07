package dagger.core

import dagger.ml.{ Instance, MultiClassClassifier }

/**
 * Created by narad on 6/19/14.
 */

abstract class Policy[D, A <: TransitionAction[S], S <: TransitionState] {}

abstract class HeuristicPolicy[D, A <: TransitionAction[S], S <: TransitionState] extends Policy[D, A, S] {

  def predict(ex: D, state: S) = chooseTransition(ex, state)

  def chooseTransition(instance: D, state: S): A

}

case class ProbabilisticClassifierPolicy[D, A <: TransitionAction[S], S <: TransitionState](classifier: MultiClassClassifier[A]) extends Policy[D, A, S] {

  def predict(ex: D, instance: Instance[A], state: S, threshold: Double = 0.0): Seq[A] = {
    if (threshold == 0.0)
      classifier.predict(instance).maxLabels
    else
      classifier.predict(instance).maxLabelsWithinThreshold(threshold)
  }

}
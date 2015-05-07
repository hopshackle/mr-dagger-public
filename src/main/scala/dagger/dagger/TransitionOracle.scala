package coref.dagger

/**
 * Created by narad on 8/26/14.
 */

trait TransitionOracle[O <: AnyRef, A <: TransitionAction[S], S <: TransitionState] {

  def acceptableActions(state: S): Seq[A]

  def chooseTransition(o: O, state: S): A
}
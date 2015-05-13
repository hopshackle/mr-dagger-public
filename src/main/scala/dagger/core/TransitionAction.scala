package dagger.core

/**
 * Created by narad on 6/19/14.
 */
abstract class TransitionAction[S <: TransitionState] {

  def apply(state: S): S

}

abstract class TransitionState {}
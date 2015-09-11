package dagger.core

/**
 * Created by narad on 10/1/14.
 */
/**
 * A transition system defines the possible paths of exploring the action state space
 * beginning at the initial state and ending in a terminal state.  It must also define
 * methods for mapping from an example of a given type D to an initial state, and a
 * definition of a terminal state
 * @tparam D the type of object being captured by the action states
 * @tparam A the type of action used by the system.  An action should map a state to a new state of the same type.
 * @tparam S the type of the state used by the system.
 */
abstract class TransitionSystem[D, A, S] {

  def actions: Array[A]

  def approximateLoss(datum: D, state: S, action: A): Double

  def chooseTransition(datum: D, state: S): A

  def construct(state: S, datum: D): D

  def init(datum: D): S

  def isPermissible(action: A, state: S): Boolean

  def isTerminal(state: S): Boolean

  def expertApprox(datum: D, state: S): D

  // A default implementation is provided, but it is recommended to override this method for greater efficiency
  def permissibleActions(state: S): Array[A] = actions.filter(action => isPermissible(action, state))

}
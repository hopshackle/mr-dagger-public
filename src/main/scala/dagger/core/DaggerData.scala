package dagger.core

abstract trait DaggerData[S <: TransitionState, A <: TransitionAction[S]] {
  
  def size: Int
  
  def getDefaultAction: A
}
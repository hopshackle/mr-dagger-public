package dagger.core

import java.io.FileWriter

import dagger.ml.Instance

import scala.collection.parallel.{ForkJoinTaskSupport, ParIterable}
import scala.concurrent.forkjoin.ForkJoinPool

// import ml.wolfe.nlp.io.ChunkReader

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Created by narad on 4/6/15.
 */
object OracleExtractor {

  def instances[D: ClassTag, A <: TransitionAction[S] : ClassTag, S <: TransitionState : ClassTag] (data: Iterable[D], trans: TransitionSystem[D, A, S],features: (D, S) => Map[Int, Double], printInterval: Int = 1000, numCores: Int = 1): Iterable[Instance[A]] = {
   // val instances = new ArrayBuffer[Instance[A]]
    val timer = new dagger.util.Timer
    timer.start()
    val instances = fork(data, numCores).flatMap { d => // } (d <- data.par) {
      // if (didx % printInterval == 0) println("Processing instance %d...".format(didx))
      var s = trans.init(d)
      val tinstances = new ArrayBuffer[Instance[A]]
      while (!trans.isTerminal(s)) {
        val permissibleActions = trans.permissibleActions(s)
        assert(permissibleActions.nonEmpty, "No permissible actions found for state:\n%s".format(s))
        val a = trans.chooseTransition(d, s)
        assert(permissibleActions.contains(a), "Oracle chose action (%s) not considered permissible by transition system for state:\n%s.".format(a, s))
        val costs = permissibleActions.map(pa => if (pa == a) 0.0 else 1.0)
        tinstances += new Instance[A](features(d, s), permissibleActions, costs)
        s = a(s)
      }
      tinstances
    }.toArray
    timer.stop()
    println("Extracted oracle instances in %s.".format(timer.toString))
    instances.toIterable
  }

  def fork[T](data: Iterable[T], forkSize: Int): ParIterable[T] = {
    val par = data.par
    par.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(forkSize))
    par
  }
}

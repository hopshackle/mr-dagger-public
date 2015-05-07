package coref.dagger

import java.io.FileWriter

import coref.ml.Instance
import ml.wolfe.nlp.io.ChunkReader

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Created by narad on 4/6/15.
 */
object OracleExtractor {


  def instances[D: ClassTag, A <: TransitionAction[S] : ClassTag, S <: TransitionState : ClassTag] (data: Iterable[D], trans: TransitionSystem[D, A, S],features: (D, S) => Map[Int, Double], printInterval: Int = 1000): Iterable[Instance[A]] = {
    val instances = new ArrayBuffer[Instance[A]]
    val timer = new coref.util.Timer
    timer.start()
    for ((d, didx) <- data.view.zipWithIndex) {
      if (didx % printInterval == 0) println("Processing instance %d...".format(didx))
      var s = trans.init(d)
      while (!trans.isTerminal(s)) {
        val permissibleActions = trans.permissibleActions(s)
        assert(permissibleActions.nonEmpty, "No permissible actions found for state:\n%s".format(s))
        val a = trans.chooseTransition(d, s)
        assert(permissibleActions.contains(a), "Oracle chose action (%s) not considered permissible by transition system for state:\n%s.".format(a, s))
        val costs = permissibleActions.map(pa => if (pa == a) 0.0 else 1.0)
        instances += new Instance[A](features(d, s), permissibleActions, costs)
        s = a(s)
      }
    }
    timer.stop()
    println("Extracted oracle instances in %s.".format(timer.toString))
    instances.toIterable
  }
}

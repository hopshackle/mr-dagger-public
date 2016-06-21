package dagger.ml

import gnu.trove._
import scala.util.Random
import gnu.trove.map.hash.THashMap

/**
 * Created by narad on 6/11/14.
 */
abstract class MultiClassClassifier[T] {

  def dotMap(v1: collection.Map[Int, Float], v2: collection.Map[Int, Float]): Float = {
    v1.foldLeft(0.0f) {
      case (sum, (f, v)) =>
        sum + v * v2.getOrElse(f, 0.0f)
    }
  }

  def predict(instance: Instance[T]): Prediction[T]

  def weightOf(a: T, p: Int): Float

  def writeToFile(filename: String, actionToString: T => String): Unit
  
  def applyAveraging: MultiClassClassifier[T] = this
}


/*
 *  The list of component classifiers assumes the most recently trained classifier is at the head of the Seq.
 *  
 */
case class SEARNClassifier[A](componentClassifiers: Seq[MultiClassClassifier[A]],
  beta: Double = 0.3, random: Random = new Random(1)) extends MultiClassClassifier[A] {

  var lastClassifierUsed = 0
  // So ... when this is initialised I can calculate the probability that any particular policy is used for a given step
  // Then when we predict we select the relevant policy stochastically, and call predict on that component classifier

  val n = componentClassifiers.size
  val tempProb = (1 to n) map (x => beta * math.pow((1 - beta), x - 1))
  val probabilities = (tempProb map (_ / tempProb.sum)).scanLeft(0.0)(_ + _).tail

  def classifierToUse: MultiClassClassifier[A] = {
    val diceRoll = random.nextDouble()
    val index = probabilities.indexWhere(diceRoll <= _)
    if (index == -1) {
      lastClassifierUsed = 0
      componentClassifiers(0)
    } else {
      lastClassifierUsed = index
      componentClassifiers(index)
    }
  }

  override def predict(instance: Instance[A]): dagger.ml.Prediction[A] = classifierToUse.predict(instance)
  override def weightOf(a: A, p: Int): Float = componentClassifiers(lastClassifierUsed).weightOf(a, p)
  override def writeToFile(filename: String, actionToString: A => String): Unit = componentClassifiers(lastClassifierUsed).writeToFile(filename, actionToString)

}
package dagger.ml

import gnu.trove._
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

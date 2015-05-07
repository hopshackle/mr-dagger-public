package coref.ml

import gnu.trove._
import gnu.trove.map.hash.THashMap

/**
 * Created by narad on 6/11/14.
 */
abstract class MultiClassClassifier[T] {

  def dotMap(v1: collection.Map[Int, Double], v2: collection.Map[Int, Double]): Double = {
    v1.foldLeft(0.0){ case(sum, (f,v)) =>
      sum + v * v2.getOrElse(f, 0.0)
    }
  }

  def predict(instance: Instance[T]): Prediction[T]

  def weightOf(a: T, p: Int): Double

  def writeToFile(filename: String): Unit
}







//  def dotMap(v1: THashMap[Int, Double], v2: THashMap[Int, Double]): Double = {
//    var sum = 0.0
//    v1.foldLeft(0.0){ case(sum, (f,v)) =>
//      sum + v * v2.getOrElse(f, 0.0)
//    }
//  }
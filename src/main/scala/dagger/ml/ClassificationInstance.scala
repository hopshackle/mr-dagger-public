package dagger.ml


import collection.Map
import scala.reflect.ClassTag

/**
 * Created with IntelliJ IDEA.
 * User: narad
 * Date: 5/23/14
 * Time: 2:43 PM
 */

case class Instance[T](feats: Map[Int, Double], labels: Array[T], costs: Array[Double] = null) {

  def featureVector = feats

  def costOf(l: T) = costs(labels.indexOf(l))

  lazy val minCost: Double = costs.head

  lazy val maxCost: Double = costs.last

  lazy val correctLabels = labels.zip(costs).filter(_._2 == 0).map(_._1)

  lazy val correctCost = 0.0

  override def toString = {
    "Instance:%s\n".format(feats.map(f => f._1 + ":" + f._2).mkString(", ")) +
      (0 until labels.size).map { i =>
        "  [%s]%s:\t%f".format(if (costOf(labels(i)) == 0.0) "+" else " ", labels(i), costs(i))
      }.mkString("\n")
  }

  def toSerialString: String = {
    feats.view.map(p => p._1).mkString(" ") + "\n" + labels.mkString(" ") + "\n" + costs.mkString(" ")
  }
}

object Instance {

  def construct[T: ClassTag](feats: Map[Int, Double], ilabels: Array[T], icosts: Array[Double], correct: Array[Boolean]): Instance[T] = {
    assert(ilabels.size > 1 && icosts.size > 1, "Insufficient costs and labels (<1) for Instance.")
    val scosts = ilabels.zip(icosts).sortBy(_._2).toArray
    var (maxCost, minCost) = (scosts.head._2, scosts.last._2)
    new Instance[T](feats, scosts.map(_._1), scosts.map(_._2 - minCost)) //, correct.zipWithIndex.filter(p => p._1).toArray.head._2)
  }

  def fromSerialString[T](str: String): Instance[T] = {
    val lines = str.split("\n")
    val feats = lines(0).split(" ").view.map(_.toInt -> 1.0).toMap
    val labels = lines(1).split(" ").map { _ match {
      case _ => "Unserializing the following action is unsupported: " + _
      }
    }.asInstanceOf[Array[T]]
    val costs = lines(2).split(" ").map(_.toDouble)
    new Instance[T](feats, labels, costs)
  }
}







//  lazy val correctCost: Double = costs(correct)

//  lazy val correctCost = 0.0

//  lazy val correctLabel: T = labels(correct)


//abstract class Instance[T](val feats: HashMap[String, Double], val labels: Array[T]) {
//
//  def featureVector = feats
//
//}
//
//class TestInstance[T](feats: HashMap[String, Double], labels: Array[T]) extends Instance[T](feats, labels) {
//
//}
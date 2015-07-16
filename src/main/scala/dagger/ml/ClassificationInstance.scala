package dagger.ml

import collection.Map
import scala.reflect.ClassTag

/**
 * Created with IntelliJ IDEA.
 * User: narad
 * Date: 5/23/14
 * Time: 2:43 PM
 */

case class Instance[T](feats: List[Map[Int, Float]], labels: Array[T], weightLabels: Array[T], costs: Array[Float] = null) {

  def featureVector = feats

  def costOf(l: T) = costs(labels.indexOf(l))

  lazy val minCost: Double = costs.head

  lazy val maxCost: Double = costs.last

  lazy val correctLabels = labels.zip(costs).filter(_._2 == 0).map(_._1)

  lazy val correctCost = 0.0

  override def toString = {
    "Instance:%s\n".format((0 until feats.size).map {
      i =>
        feats(i) map (f => f._1 + ":" + f._2) mkString (", ") +
          "  [%s]%s:\t%f".format(if (costOf(labels(i)) == 0.0) "+" else " ", labels(i), costs(i))
    })
  }

  def toSerialString: String = {
    feats.size + "\n" +
      (feats map (f => f.view.map(p => p._1).mkString(" ") + "\n")) +
      labels.mkString(" ") + "\n" + costs.mkString(" ")
  }
}

object Instance {

  def construct[T: ClassTag](feats: List[Map[Int, Float]], ilabels: Array[T], icosts: Array[Float], correct: Array[Boolean]): Instance[T] = {
    assert(ilabels.size > 1 && icosts.size > 1, "Insufficient costs and labels (<1) for Instance.")
    val scosts = (ilabels, icosts, feats).zipped.toList.sortBy(_._2)
    var (maxCost, minCost) = (scosts.head._2, scosts.last._2)
    new Instance[T](scosts.map(_._3), scosts.map(_._1).toArray, scosts.map(_._1).toArray, scosts.map(_._2 - minCost).toArray) //, correct.zipWithIndex.filter(p => p._1).toArray.head._2)
  }

  def fromSerialString[T](str: String): Instance[T] = {
    val lines = str.split("\n")
    val number = lines(0).toInt
    val feats = ((1 to number) map (i => lines(i).split(" ").view.map(_.toInt -> 1.0f).toMap)).toList
    val labels = lines(number+1).split(" ").map {
      _ match {
        case _ => "Unserializing the following action is unsupported: " + _
      }
    }.asInstanceOf[Array[T]]
    val costs = lines(number+2).split(" ").map(_.toFloat)
    new Instance[T](feats, labels, labels, costs)
  }
}

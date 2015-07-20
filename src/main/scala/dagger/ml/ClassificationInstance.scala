package dagger.ml

import collection.Map
import scala.reflect.ClassTag
import gnu.trove.map.hash.THashMap

/**
 * Created with IntelliJ IDEA.
 * User: narad
 * Date: 5/23/14
 * Time: 2:43 PM
 */

case class Instance[T](feats: List[gnu.trove.map.hash.THashMap[Int, Float]], labels: Array[T], weightLabels: Array[T], costs: Array[Float] = null) {

  def featureVector = feats map (i => Instance.troveMapToScala(i))

  def costOf(l: T) = costs(labels.indexOf(l))

  lazy val minCost: Double = costs.head

  lazy val maxCost: Double = costs.last

  lazy val correctLabels = labels.zip(costs).filter(_._2 == 0).map(_._1)

  lazy val correctCost = 0.0

  override def toString = {
    "Instance:%s\n".format((0 until feats.size).map {
      i =>
        Instance.troveMapToScala(feats(i)) map (f => f._1 + ":" + f._2) mkString (", ") +
          "  [%s]%s:\t%f".format(if (costOf(labels(i)) == 0.0) "+" else " ", labels(i), costs(i))

    })
  }

}

object Instance {

  def construct[T: ClassTag](feats: List[gnu.trove.map.hash.THashMap[Int, Float]], ilabels: Array[T], icosts: Array[Float], correct: Array[Boolean]): Instance[T] = {
    assert(ilabels.size > 1 && icosts.size > 1, "Insufficient costs and labels (<1) for Instance.")
    val scosts = (ilabels, icosts, feats).zipped.toList.sortBy(_._2)
    var (maxCost, minCost) = (scosts.head._2, scosts.last._2)
    new Instance[T](scosts.map(_._3), scosts.map(_._1).toArray, scosts.map(_._1).toArray, scosts.map(_._2 - minCost).toArray) //, correct.zipWithIndex.filter(p => p._1).toArray.head._2)
  }

  def troveMapToScala(trove: gnu.trove.map.hash.THashMap[Int, Float]): Map[Int, Float] = {
    import scala.collection.JavaConversions._
    trove
  }

  def scalaMapToTrove(scalaMap: Map[Int, Float]): gnu.trove.map.hash.THashMap[Int, Float] = {
    var output = new gnu.trove.map.hash.THashMap[Int, Float]()
    for (key <- scalaMap.keys) {
      output.put(key, scalaMap(key))
    }
    output
  }

}

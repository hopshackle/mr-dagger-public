package dagger.ml

import collection.Map
import scala.reflect.ClassTag
import gnu.trove.map.hash._
import gnu.trove.procedure._

/**
 * Created with IntelliJ IDEA.
 * User: narad
 * Date: 5/23/14
 * Time: 2:43 PM
 */
// A <: TransitionAction[S]: ClassTag, S <: TransitionState: ClassTag
case class Instance[T](feats: List[gnu.trove.map.hash.THashMap[Int, Float]], labels: Array[T], weightLabels: Array[T], 
    costs: Array[Float] = null, err: Int = 0) {

  lazy val featureVector = feats map (i => Instance.troveMapToScala(i))

  def costOf(l: T) = costs(labels.indexOf(l))

  lazy val minCost: Double = costs.head

  lazy val maxCost: Double = costs.last

  lazy val correctLabels = labels.zip(costs).filter(_._2 == 0).map(_._1)

  lazy val correctCost = 0.0

  var errors = err

  def fileFormat(actionToString: (T => String)): String = {
    val actionSize = feats.size + "\n"
    val errorCount = getErrorCount + "\n"
    val featureOutput = ((0 until feats.size) map (i => featureVector(i) map (f => f"${f._1}:${f._2}%.2f") mkString ("\t")) mkString ("\n")) + "\n"
    val labelOutput = (labels map actionToString mkString ("\t")) + "\n"
    val weightLabelOutput = (weightLabels map actionToString mkString ("\t")) + "\n"
    val costOutput = (costs map { c => f"${c}%.4f" } mkString ("\t")) + "\n"
    actionSize + errorCount + featureOutput + labelOutput + weightLabelOutput + costOutput + "END\n"
  }

  def errorIncrement: Unit = {
    errors += 1
  }
  def getErrorCount: Int = errors

  // This method is not used .. it is slightly faster and more elegant, but does not make a copy, and updates fileCache in an unwanted fashion
  def removeFeaturesII(toRemove: Set[Int]): Instance[T] = {
    for (i <- 0 until feats.size) {
      feats(i).retainEntries(new TObjectObjectProcedure[Int, Float]() {
        def execute(k: Int, v: Float): Boolean = {
          !(toRemove contains k)
        }
      })
    }
    this
  }

  override def toString = {
    "Instance:%s\n".format((0 until feats.size).map {
      i =>
        Instance.troveMapToScala(feats(i)) map (f => f._1 + ":" + f._2) mkString (", ") +
          "  [%s]%s:\t%f".format(if (costOf(labels(i)) == 0.0) "+" else " ", labels(i), costs(i))

    })
  }

}

object Instance {

  var rareFeatures = Set[Int]()
  def setRareFeatures(feat: Set[Int]): Unit = rareFeatures = feat

  def construct[T: ClassTag](feats: List[gnu.trove.map.hash.THashMap[Int, Float]], ilabels: Array[T], icosts: Array[Float], correct: Array[Boolean]): Instance[T] = {
    assert(ilabels.size > 1 && icosts.size > 1, "Insufficient costs and labels (<1) for Instance.")
    val scosts = (ilabels, icosts, feats).zipped.toList.sortBy(_._2)
    var (maxCost, minCost) = (scosts.head._2, scosts.last._2)
    new Instance[T](scosts.map(_._3), scosts.map(_._1).toArray, scosts.map(_._1).toArray, scosts.map(_._2 - minCost).toArray) //, correct.zipWithIndex.filter(p => p._1).toArray.head._2)
  }

  def construct[T: ClassTag](input: String, stringToAction: (String => T)): Instance[T] = {
    // input is in the same format as that constructed by fileFormat function
    val inputSplit = input.split("\n")
    val actionSize = inputSplit(0).toInt
    val errors = inputSplit(1).toInt
    val features = ((2 to actionSize + 1) map (i =>
      scalaMapToTrove((inputSplit(i).split("\t") map { t => (t.split(":")(0).toInt, t.split(":")(1).toFloat) }).toMap))).toList
    val labels = inputSplit(actionSize + 2).split("\t") map stringToAction
    val weightLabels = inputSplit(actionSize + 3).split("\t") map stringToAction
    val costs = inputSplit(actionSize + 4).split("\t") map { i => i.toFloat }
    new Instance[T](features, labels, weightLabels, costs, errors)
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

  def pruneRareFeatures(feats: THashMap[Int, Float]): THashMap[Int, Float] = {
    import scala.collection.JavaConversions._
    var prunedFeatures = new THashMap[Int, Float]()
    for (j <- feats.keys) {
      if (rareFeatures contains j) {
        // do nothing
      } else {
        prunedFeatures.put(j, feats.get(j))
      }
    }
    prunedFeatures
  }

}

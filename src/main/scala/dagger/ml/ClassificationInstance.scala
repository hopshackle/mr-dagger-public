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
case class Instance[T](coreFeats: gnu.trove.map.hash.THashMap[Int, Float], parameterFeats: Map[Int, gnu.trove.map.hash.THashMap[Int, Float]],
  labels: Array[T], weightLabels: Array[T], costs: Array[Float] = null, err: Int = 0) {

  private val mergedFeatures = scala.collection.mutable.Map[Int, gnu.trove.map.hash.THashMap[Int, Float]]()

  import Instance.troveMapToScala
  lazy val coreFeatsScala = troveMapToScala(coreFeats)
  //  lazy val featureVector = {
  //    labels.zipWithIndex map { case (l, i) => coreFeatsScala ++ troveMapToScala(parameterFeats(i)) }
  //  }

  def featureVector(labelRef: Int): Map[Int, Float] = {
    coreFeatsScala ++ (if (parameterFeats.contains(labelRef)) troveMapToScala(parameterFeats(labelRef)) else Map())
  }
  def feats(labelRef: Int): gnu.trove.map.hash.THashMap[Int, Float] = {
    if (!parameterFeats.contains(labelRef))
      coreFeats
    else {
      if (mergedFeatures.contains(labelRef))
        mergedFeatures(labelRef)
      else {
        val output = new gnu.trove.map.hash.THashMap[Int, Float]()
        output.putAll(coreFeats)
        output.putAll(parameterFeats(labelRef))
        mergedFeatures(labelRef) = output
        output
      }
    }
  }

  def costOf(l: T) = costs(labels.indexOf(l))

  lazy val minCost: Double = costs.head

  lazy val maxCost: Double = costs.last

  lazy val correctLabels = labels.zip(costs).filter(_._2 == 0).map(_._1)

  lazy val correctCost = 0.0

  var errors = err

  def fileFormat(actionToString: (T => String)): String = {
    def featureToString(f: (Int, Float)): String = f"${f._1}:${f._2}%.2f"
    def featureMapToString(m: Map[Int, Float]): String = (m map featureToString mkString ("\t")) + "\n"
    val actionSize = labels.size + "\n"
    val errorCount = getErrorCount + "\n"
    val coreFeatureOutput = featureMapToString(coreFeatsScala)
    val parameterFeatureOutput = (parameterFeats map { case (k, v) => k.toString + "\t" + featureMapToString(troveMapToScala(v)) }).mkString("")
    val labelOutput = (labels map actionToString mkString ("\t")) + "\n"
    val weightLabelOutput = (weightLabels map actionToString mkString ("\t")) + "\n"
    val costOutput = (costs map { c => f"${c}%.4f" } mkString ("\t")) + "\n"
    actionSize + errorCount + coreFeatureOutput + parameterFeatureOutput + "END FEATURES\n" +
      labelOutput + weightLabelOutput + costOutput + "END\n"
  }

  def errorIncrement: Unit = {
    errors += 1
  }
  def getErrorCount: Int = errors

  override def toString = {
    "Instance:%s\n".format((labels map (_.toString)) mkString (" : "))
  }

}

object Instance {

  var rareFeatures = Set[Int]()
  val dummyFeatures = (-1 -> (new gnu.trove.map.hash.THashMap[Int, Float]()))
  def setRareFeatures(feat: Set[Int]): Unit = rareFeatures = feat

  def construct[T: ClassTag](coreFeats: gnu.trove.map.hash.THashMap[Int, Float],
    paramFeats: Map[Int, gnu.trove.map.hash.THashMap[Int, Float]],
    ilabels: Array[T], icosts: Array[Float], correct: Array[Boolean]): Instance[T] = {
    assert(ilabels.size > 1 && icosts.size > 1, "Insufficient costs and labels (<1) for Instance.")
    val labelsCostsInCostOrder = (ilabels, icosts).zipped.toList.sortBy(_._2)
    var (maxCost, minCost) = (labelsCostsInCostOrder.head._2, labelsCostsInCostOrder.last._2)
    new Instance[T](coreFeats, paramFeats, labelsCostsInCostOrder.map(_._1).toArray,
      labelsCostsInCostOrder.map(_._1).toArray, labelsCostsInCostOrder.map(_._2 - minCost).toArray)
  }

  def construct[T: ClassTag](input: String, stringToAction: (String => T)): Instance[T] = {
    // input is in the same format as that constructed by fileFormat function
    def featuresToTroveMap(rawFeatures: String): gnu.trove.map.hash.THashMap[Int, Float] = {
      scalaMapToTrove((rawFeatures.split("\t") map { t => (t.split(":")(0).toInt, t.split(":")(1).toFloat) }).toMap)
    }
    import scala.language.postfixOps
    def featuresToIndexAndTroveMap(rawFeatures: String): (Int, gnu.trove.map.hash.THashMap[Int, Float]) = {
      val splitFeatures = rawFeatures.split("\t")
      val troveMap = scalaMapToTrove(splitFeatures.drop(1) map { t => (t.split(":")(0).toInt, t.split(":")(1).toFloat) } toMap)
 //     println(troveMap)
      (splitFeatures(0).toInt, troveMap)
    }
    val inputSplit = input.split("\n")
    val actionSize = inputSplit(0).toInt
    val errors = inputSplit(1).toInt
    val coreFeatures = featuresToTroveMap(inputSplit(2))

    var endFeatures = false
    val parameterFeatures = ((3 until (3 + actionSize)) map { lineNumber =>
      if (!endFeatures) {
        val nextLine = inputSplit(lineNumber)
        if (nextLine == "END FEATURES") {
          endFeatures = true
          dummyFeatures
        } else {
          featuresToIndexAndTroveMap(nextLine)
        }
      } else {
        dummyFeatures
      }
    }) toMap

    val cleanedParameterFeatures = parameterFeatures filterNot { case (k, v) => v.isEmpty }
    val parameterLines = cleanedParameterFeatures.size
    val labels = inputSplit(4 + parameterLines).split("\t") map stringToAction
    val weightLabels = inputSplit(5 + parameterLines).split("\t") map stringToAction
    val costs = inputSplit(6 + parameterLines).split("\t") map { i => i.toFloat }
    new Instance[T](coreFeatures, cleanedParameterFeatures, labels, weightLabels, costs, errors)
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

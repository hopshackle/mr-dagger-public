package dagger.ml

import collection.Map
import scala.collection.mutable.HashMap
import scala.reflect.ClassTag

/**
 * Created with IntelliJ IDEA.
 * User: narad
 * Date: 5/23/14
 * Time: 2:43 PM
 */
// A <: TransitionAction[S]: ClassTag, S <: TransitionState: ClassTag
case class Instance[T](coreFeats: Map[Int, Float], parameterFeats: Map[Int, Map[Int, Float]],
  labels: Array[T], weightLabels: Array[T], costs: Array[Float] = null, err: Int = 0) {

  private val mergedFeatures = HashMap[Int, Map[Int, Float]]()

  def feats(labelRef: Int): Map[Int, Float] = {
    if (!parameterFeats.contains(labelRef))
      coreFeats
    else {
      if (!mergedFeatures.contains(labelRef))
        mergedFeatures(labelRef) = coreFeats ++ parameterFeats(labelRef)
      mergedFeatures(labelRef)
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
    val coreFeatureOutput = featureMapToString(coreFeats)
    val parameterFeatureOutput = (parameterFeats map { case (k, v) => k.toString + "\t" + featureMapToString(v) }).mkString("")
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
  val dummyFeatures = (-1 -> (Map[Int, Float]()))
  def setRareFeatures(feat: Set[Int]): Unit = rareFeatures = feat

  def construct[T: ClassTag](coreFeats: Map[Int, Float],
    paramFeats: Map[Int, Map[Int, Float]],
    ilabels: Array[T], icosts: Array[Float], correct: Array[Boolean]): Instance[T] = {
    assert(ilabels.size > 1 && icosts.size > 1, "Insufficient costs and labels (<1) for Instance.")
    val labelsCostsInCostOrder = (ilabels, icosts).zipped.toList.sortBy(_._2)
    var (maxCost, minCost) = (labelsCostsInCostOrder.head._2, labelsCostsInCostOrder.last._2)
    new Instance[T](coreFeats, paramFeats, labelsCostsInCostOrder.map(_._1).toArray,
      labelsCostsInCostOrder.map(_._1).toArray, labelsCostsInCostOrder.map(_._2 - minCost).toArray)
  }

  def construct[T: ClassTag](input: String, stringToAction: (String => T)): Instance[T] = {
    // input is in the same format as that constructed by fileFormat function
    def featuresToMap(rawFeatures: String): Map[Int, Float] = {
      (rawFeatures.split("\t") map { t => (t.split(":")(0).toInt, t.split(":")(1).toFloat) }).toMap
    }
    def featuresToIndexAndMap(rawFeatures: String): (Int, Map[Int, Float]) = {
      val splitFeatures = rawFeatures.split("\t")
      val mapForm = (splitFeatures.drop(1) map { t => (t.split(":")(0).toInt, t.split(":")(1).toFloat) }).toMap
      (splitFeatures(0).toInt, mapForm)
    }
    val inputSplit = input.split("\n")
    val actionSize = inputSplit(0).toInt
    val errors = inputSplit(1).toInt
    val coreFeatures = featuresToMap(inputSplit(2))

    var endFeatures = false
    val parameterFeatures = ((3 until (3 + actionSize)) map { lineNumber =>
      if (!endFeatures) {
        val nextLine = inputSplit(lineNumber)
        if (nextLine == "END FEATURES") {
          endFeatures = true
          dummyFeatures
        } else {
          featuresToIndexAndMap(nextLine)
        }
      } else {
        dummyFeatures
      }
    })

    val filteredParameterFeatures = parameterFeatures filterNot (_._1 == -1)
    val parameterLines = filteredParameterFeatures.size

    val labels = inputSplit(parameterLines + 4).split("\t") map stringToAction
    val weightLabels = inputSplit(parameterLines + 5).split("\t") map stringToAction
    val costs = inputSplit(parameterLines + 6).split("\t") map { i => i.toFloat }
    new Instance[T](coreFeatures, filteredParameterFeatures.toMap, labels, weightLabels, costs, errors)
  }

  def pruneRareFeatures(feats: Map[Int, Float]): Map[Int, Float] = {
    import scala.collection.JavaConversions._
    var prunedFeatures = new HashMap[Int, Float]()
    for (j <- feats.keys) {
      if (rareFeatures contains j) {
        // do nothing
      } else {
        prunedFeatures.put(j, feats(j))
      }
    }
    prunedFeatures
  }

}

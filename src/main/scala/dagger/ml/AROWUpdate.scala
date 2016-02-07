package dagger.ml
import scala.reflect.ClassTag
import scala.collection.Map
import scala.collection.mutable.HashMap

class AROWUpdate[T: ClassTag] extends PerceptronUpdateRule {

   override def update(instance: Instance[T], classifier: AROWClassifier[T], options: AROWOptions): Unit = {
    val prediction = classifier.predict(instance)
    val temp = instance.correctLabels.map(l => (l, prediction.label2score(l))).toArray.sortBy(_._2)
    if (temp.isEmpty) println("No Correct Labels found for: \n" + instance)
    val weightVectors = classifier.weights
    val cachedWeightVectors = classifier.cachedWeights
    val varianceVectors = classifier.variances
    val maxLabel = prediction.maxLabel
    val maxScore = prediction.maxScore
    val (minCorrectLabel, minCorrectScore) = temp.head
    val labelList = instance.labels
    val iMaxLabel = labelList.indexOf(maxLabel)
    val icost = instance.costOf(maxLabel)
    val maxWeightLabel = instance.weightLabels(iMaxLabel)
    val iMinCorrectLabel = labelList.indexOf(minCorrectLabel)
    val minCorrectWeightLabel = instance.weightLabels(iMinCorrectLabel)

    val zVectorPredicted = new HashMap[Int, Float]()
    val zVectorMinCorrect = new HashMap[Int, Float]()
    for (feat <- (instance.feats(iMaxLabel).keySet diff Instance.rareFeatures)) {
      if (varianceVectors.contains(maxWeightLabel))
        zVectorPredicted(feat) = instance.feats(iMaxLabel)(feat) * varianceVectors(maxWeightLabel).getOrElse(feat, 1.0f)
      else
        zVectorPredicted(feat) = instance.feats(iMaxLabel)(feat)
    }
    for (feat <- (instance.feats(iMinCorrectLabel).keySet diff Instance.rareFeatures)) {
      if (varianceVectors.contains(minCorrectWeightLabel))
        zVectorMinCorrect(feat) = instance.feats(iMinCorrectLabel)(feat) * varianceVectors(minCorrectWeightLabel).getOrElse(feat, 1.0f)
      else
        zVectorMinCorrect(feat) = instance.feats(iMinCorrectLabel)(feat)
    }

    val preDot = AROW.dotMap(instance.feats(iMaxLabel), zVectorPredicted)
    val minDot = AROW.dotMap(instance.feats(iMinCorrectLabel), zVectorMinCorrect)
    val confidence = preDot + minDot

    val beta = 1.0f / (confidence + options.SMOOTHING.toFloat)
    val loss = (maxScore - minCorrectScore + Math.sqrt(icost)).toFloat
    val alpha = loss * beta

    if (options.VERBOSE) {
      println("confidence = " + confidence)
      println("max label = " + maxLabel)
      println("max score = " + maxScore)
      println("correct label = " + minCorrectLabel)
      println("correct score = " + minCorrectScore)
      println("Instance cost of max prediction = " + icost)
      println("alpha = " + alpha)
      println("beta = " + beta)
      println("loss = " + loss)
      println("pre dot = " + preDot)
      println("min dot = " + minDot)
    }

    AROW.add(weightVectors(maxWeightLabel), zVectorPredicted, -1.0f * alpha)
    AROW.add(weightVectors(minCorrectWeightLabel), zVectorMinCorrect, alpha)
    if (options.AVERAGING) {
      AROW.add(cachedWeightVectors(maxWeightLabel), zVectorPredicted, -1.0f * alpha * classifier.averagingCounter)
      AROW.add(cachedWeightVectors(minCorrectWeightLabel), zVectorMinCorrect, alpha * classifier.averagingCounter)
    }

    for (feat <- instance.feats(iMaxLabel).keySet diff Instance.rareFeatures) {
      // AV: you can save yourself this if by initializing them in the beginning
      if (!varianceVectors.contains(maxWeightLabel)) varianceVectors(maxWeightLabel) = new HashMap[Int, Float]
      varianceVectors(maxWeightLabel)(feat) = varianceVectors(maxWeightLabel).getOrElse(feat, 1.0f) - beta * math.pow(zVectorPredicted(feat), 2).toFloat
    }
    for (feat <- instance.feats(iMinCorrectLabel).keySet diff Instance.rareFeatures) {
      // AV: you can save yourself this if by initializing them in the beginning
      if (!varianceVectors.contains(minCorrectWeightLabel)) varianceVectors(minCorrectWeightLabel) = new HashMap[Int, Float]
      varianceVectors(minCorrectWeightLabel)(feat) = (varianceVectors(minCorrectWeightLabel).getOrElse(feat, 1.0f) - beta * math.pow(zVectorMinCorrect(feat), 2)).toFloat
    }
  }
}
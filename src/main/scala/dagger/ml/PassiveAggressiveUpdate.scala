package dagger.ml
import scala.reflect.ClassTag
import scala.collection.Map
import scala.collection.mutable.HashMap

class PassiveAggressiveUpdate[T: ClassTag] extends PerceptronUpdateRule {

  override def update(instance: Instance[T], classifier: AROWClassifier[T], options: AROWOptions): Unit = {
    val prediction = classifier.predict(instance)
    val temp = instance.correctLabels.map(l => (l, prediction.label2score(l))).toArray.sortBy(_._2)
    if (temp.isEmpty) println("No Correct Labels found for: \n" + instance)
    val weightVectors = classifier.weights
    val cachedWeightVectors = classifier.cachedWeights
    val maxLabel = prediction.maxLabel
    val maxScore = prediction.maxScore
    val (minCorrectLabel, minCorrectScore) = temp.head
    val labelList = instance.labels
    val iMaxLabel = labelList.indexOf(maxLabel)
    val icost = instance.costOf(maxLabel)
    val maxWeightLabel = instance.weightLabels(iMaxLabel)
    val iMinCorrectLabel = labelList.indexOf(minCorrectLabel)
    val minCorrectWeightLabel = instance.weightLabels(iMinCorrectLabel)
    val loss = (maxScore - minCorrectScore + math.sqrt(instance.costOf(maxLabel))).toFloat
    val norm = 2 * (AROW.dotMap(instance.feats(iMinCorrectLabel), instance.feats(iMinCorrectLabel)))
    val factor = loss / (norm + (1.0f / (2 * options.SMOOTHING.toFloat)))

    AROW.add(weightVectors(maxWeightLabel), instance.feats(iMaxLabel), -1.0f * factor)
    AROW.add(weightVectors(minCorrectWeightLabel), instance.feats(iMinCorrectLabel), factor)
    if (options.AVERAGING) {
      AROW.add(cachedWeightVectors(maxWeightLabel), instance.feats(iMaxLabel), -1.0f * factor * classifier.averagingCounter)
      AROW.add(cachedWeightVectors(minCorrectWeightLabel), instance.feats(iMinCorrectLabel), factor * classifier.averagingCounter)
    }
  }
}
package dagger.ml

import java.io.FileWriter

import scala.collection.Map
import scala.collection.mutable.HashMap
import scala.reflect.ClassTag
import gnu.trove._
import scala.util.Random

/**
 * Created by narad on 4/6/15.
 */
class PassiveAggressiveClassifier[T](val weights: HashMap[T, HashMap[Int, Float]]) extends MultiClassClassifier[T] {

  def predict(instance: Instance[T]): Prediction[T] = {
    val scores: Map[T, Float] = if (weights.isEmpty) {
      instance.labels.map { label => label -> 1.0f }.toMap
    } else {
      ((instance.labels.toList.zipWithIndex) map {
        case (label, i) =>
          label -> dotMap(instance.featureVector(i), weights(label))
      }).toMap
    }
    Prediction[T](label2score = scores)
  }

  def weightOf(a: T, p: Int): Float = {
    if (weights.contains(a)) {
      weights(a).get(p).get
    } else {
      0.0f
    }
  }

  def writeToFile(filename: String) = {
    val out = new FileWriter(filename)
    for (label <- weights.keys; (f, w) <- weights(label)) {
      out.write(label + "\t" + f + "\t" + w + "\n")
    }
    out.close()
  }
}

object PassiveAggressive {

  def train[T: ClassTag](data: Iterable[Instance[T]], labels: Seq[T], rate: Double = 0.1, random: Random, options: AROWOptions, verbose: Boolean = false): PassiveAggressiveClassifier[T] = {
    val smoothing = options.SMOOTHING.toFloat
    // Initialize weight and variance vectors
    val weightVectors = new HashMap[T, HashMap[Int, Float]]
    for (label <- labels) weightVectors.put(label, new HashMap[Int, Float])

    // Begin training loop
    val rounds = options.TRAIN_ITERATIONS
    println("Beginning %d rounds of perceptron training with %d instances and leaning rate of parameter %.2f.".format(rounds, data.size, rate))
    var classifier = new PassiveAggressiveClassifier[T](weightVectors)
    for (r <- 1 to rounds) {
      val instances = data
      var errors = 0.0
      var cost = 0.0
      var icount = 0
      for (instance <- instances) {
        if (options.AROW_PRINT_INTERVAL > 0 && icount % options.AROW_PRINT_INTERVAL == 0) print("\rRound %d...instance %d...".format(r, icount))
        icount += 1

        val prediction = classifier.predict(instance)
        val maxLabel = prediction.randomMaxLabel(random)
        val maxScore = prediction.maxScore
        val icost = instance.costOf(maxLabel)
        if (instance.costOf(maxLabel) > 0) {
          errors += 1
          cost += instance.costOf(maxLabel)

          var minCorrectScore = instance.correctCost
          var minCorrectLabel = instance.correctLabels.head
          val labelList = instance.labels

          for (label <- Array(minCorrectLabel)) {
            val iLabel = labelList.indexOf(label)
            val score = dotMap(instance.featureVector(iLabel), weightVectors(label))
            if (score < minCorrectScore) {
              minCorrectScore = score
              minCorrectLabel = label
            }
          }

          val iMinCorrectLabel = labelList.indexOf(minCorrectLabel)
          val loss = (maxScore - minCorrectScore + math.sqrt(instance.costOf(maxLabel))).toFloat
          val norm = 2 * (dotMap(instance.featureVector(iMinCorrectLabel), instance.featureVector(iMinCorrectLabel)))
          val factor = loss / (norm + (1.0f / (2 * smoothing)))

          val iMaxLabel = labelList.indexOf(maxLabel)
          add(weightVectors(maxLabel), instance.featureVector(iMaxLabel), -1.0f * factor)
          add(weightVectors(minCorrectLabel), instance.featureVector(iMinCorrectLabel), factor)
        }
      }
      classifier = new PassiveAggressiveClassifier[T](weightVectors)
      if (verbose) println("Training error in round %d : %1.2f".format(r, (100 * errors / data.size)))
    }

    // Compute final training error
    var finalErrors = 0.0
    var finalCost = 0.0
    for (instance <- data) {

      val prediction = classifier.predict(instance)
      val maxLabel = prediction.randomMaxLabel(random)
      val maxCost = instance.costOf(maxLabel)
      if (maxCost > 0) {
        finalErrors += 1
        finalCost += maxCost
      }
    }
    println("Final training error rate (%1.0f / %d) = %1.3f".format(finalErrors, data.size, 100 * finalErrors / data.size))
    println("Final training cost = %1.3f".format(finalErrors))

    // Return final classifier
    new PassiveAggressiveClassifier[T](weightVectors)
  }

  def add(v1: HashMap[Int, Float], v2: collection.Map[Int, Float], damp: Float = 1.0f): Unit = {
    for ((key, value) <- v2) v1(key) = v1.getOrElse(key, 0.0f) + value * damp
  }

  def dotMap(v1: collection.Map[Int, Float], v2: collection.Map[Int, Float]): Float = {
    v1.foldLeft(0.0f) { case (sum, (f, v)) => sum + v * v2.getOrElse(f, 0.0f) }
  }

}
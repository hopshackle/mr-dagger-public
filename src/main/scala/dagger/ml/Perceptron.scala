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
class PerceptronClassifier[T](val weights: HashMap[T, HashMap[Int, Double]]) extends MultiClassClassifier[T] {

  def predict(instance: Instance[T]): Prediction[T] = {
    val scores: Map[T, Double] = if (weights.isEmpty) {
      instance.labels.map { label => label -> 1.0 }.toMap
    }
    else {
      instance.labels.map { label =>
        label -> dotMap(instance.featureVector, weights(label))
      }.toMap
    }
    Prediction[T](label2score = scores)
  }

  def weightOf(a: T, p: Int): Double = {
    if (weights.contains(a)) {
      weights(a).get(p).get
    }
    else {
      0.0
    }
  }

  def writeToFile(filename: String) = {
    val out = new FileWriter(filename)
    for (label <- weights.keys; (f,w) <- weights(label)) {
      out.write(label + "\t" + f + "\t" + w + "\n")
    }
    out.close()
  }
}




object Perceptron {


  def train[T: ClassTag](data: Iterable[Instance[T]], labels: Seq[T], rate: Double = 0.1, random: Random, options: AROWOptions, verbose: Boolean = false): PerceptronClassifier[T] = {
    // Initialize weight and variance vectors
    val weightVectors = new HashMap[T, HashMap[Int, Double]]
    for (label <- labels) weightVectors.put(label, new HashMap[Int, Double])

    // Begin training loop
    val rounds = options.TRAIN_ITERATIONS
    println("Beginning %d rounds of perceptron training with %d instances and leaning rate of parameter %.2f.".format(rounds, data.size, rate))
    var classifier = new PerceptronClassifier[T](weightVectors)
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
          for (label <- Array(minCorrectLabel)) {
            val score = dotMap(instance.featureVector, weightVectors(label))
            if (score < minCorrectScore) {
              minCorrectScore = score
              minCorrectLabel = label
            }
          }


          val zVectorPredicted = new HashMap[Int, Double]()
          val zVectorMinCorrect = new HashMap[Int, Double]()

          val preDot = dotMap(instance.featureVector, zVectorPredicted)
          val minDot = dotMap(instance.featureVector, zVectorMinCorrect)

          val confidence = preDot + minDot

          val loss = maxScore - minCorrectScore + math.sqrt(instance.costOf(maxLabel))
          val alpha = loss * rate


          add(weightVectors(maxLabel), zVectorPredicted, -1.0 * alpha)
          add(weightVectors(minCorrectLabel), zVectorMinCorrect, alpha)

        }
      }
      classifier = new PerceptronClassifier[T](weightVectors)
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
    new PerceptronClassifier[T](weightVectors)
  }

  def add(v1: HashMap[Int, Double], v2: HashMap[Int, Double], damp: Double = 1.0) = {
    for ((key,value) <- v2) v1(key) = v1.getOrElse(key,0.0) + value * damp
  }

  def dotMap(v1: collection.Map[Int, Double], v2: collection.Map[Int, Double]): Double = {
    v1.foldLeft(0.0){ case(sum, (f,v)) => sum + v * v2.getOrElse(f, 0.0)}
  }

}
package dagger.ml

import java.io.{ File, FileWriter }

import collection.mutable.HashMap
// import java.util.HashMap
import scala.reflect.ClassTag
import scala.util.Random
//import collection.immutable.Map
//import scala.collection.JavaConversions.mapAsScalaMap

/**
 * Created with IntelliJ IDEA.
 * User: narad
 * Date: 5/14/14
 * Time: 12:13 PM
 */
case class AROWClassifier[T: ClassTag](weights: HashMap[T, HashMap[Int, Double]] = new HashMap[T, HashMap[Int, Double]](),
  variances: HashMap[T, HashMap[Int, Double]] = new HashMap[T, HashMap[Int, Double]]())
  extends MultiClassClassifier[T] {

  def predict(instance: Instance[T]): Prediction[T] = {
    //  println("instance is null ?" + instance == null)
    //  println("size:" + instance.labels.size)

    val scores = (instance.labels, instance.weightLabels, instance.featureVector).zipped map {
      case (label, weightLabel, feats) =>
        if (!weights.contains(weightLabel)) weights(weightLabel) = new HashMap[Int, Double]
        label -> dotMap(feats, weights(weightLabel))
    }
    Prediction[T](label2score = scores.toMap)
  }

  def weightOf(a: T, p: Int): Double = weights(a).getOrElse(p, 0.0)

  def writeToFile(filename: String) = {
    println("Labels: " + weights.keys.mkString(", "))
    val file = new File(filename)
    if (!file.getParentFile.exists()) file.getParentFile.mkdirs()
    val out = new FileWriter(filename)
    for (label <- weights.keys; (f, w) <- weights(label)) {
      out.write(label + "\t" + f + "\t" + w + "\n")
    }
    out.close()
  }
}

object AROWClassifier {

  def empty[T: ClassTag](labels: Array[T] = Array(), weightLabels: Array[T]): AROWClassifier[T] = {
    val weights = new HashMap[T, HashMap[Int, Double]]
    val variances = new HashMap[T, HashMap[Int, Double]]
    for (l <- weightLabels) {
      if (!weights.contains(l)) weights(l) = new HashMap[Int, Double]
      if (!variances.contains(l)) variances(l) = new HashMap[Int, Double]
    }
    AROWClassifier(weights, variances)
  }

  def fromFile[T: ClassTag](filename: String, actionMap: (String => T)): AROWClassifier[T] = {
    val weights = new HashMap[T, HashMap[Int, Double]]
    val variances = new HashMap[T, HashMap[Int, Double]]
    val lines = io.Source.fromFile(filename).getLines
    lines.foreach { line =>
      val cols = line.split("\t")
      val label = actionMap(cols(0))
      val f = cols(1).toInt
      val w = cols(2).toDouble
      if (!weights.contains(label)) weights(label) = new HashMap[Int, Double]
      weights(label)(f) = w
    }
    AROWClassifier(weights, variances)
  }
}

object AROW {

  def train[T: ClassTag](data: Iterable[Instance[T]], labels: Array[T], weightLabels: Array[T], options: AROWOptions, init: Option[AROWClassifier[T]] = None): AROWClassifier[T] = {
    val pruned = removeRareFeatures(data, options.RARE_FEATURE_COUNT)
    val random = new Random(options.RANDOM_SEED)
    val model = init match {
      case Some(arow) => arow
      case None => AROWClassifier.empty[T](labels = labels, weightLabels = weightLabels)
    }
    val smoothing = if (options.TUNE_REGULARIZER) {
      tuneSmoothingParameter(data = pruned, labels = labels, weightLabels = weightLabels, init = model, random)
    } else {
      options.SMOOTHING
    }
    trainFromClassifier(
      data = pruned,
      rounds = options.TRAIN_ITERATIONS,
      shuffle = options.SHUFFLE,
      averaging = options.AVERAGING,
      smoothing = smoothing,
      init = model,
      random = random)
  }

  private def trainFromClassifier[T: ClassTag](data: Iterable[Instance[T]], rounds: Int, smoothing: Double = 1.0, shuffle: Boolean = true,
    averaging: Boolean = true, init: AROWClassifier[T], printInterval: Int = 100000, verbose: Boolean = false, random: Random): AROWClassifier[T] = {
    // Initialize weight and variance vectors
    val weightVectors = init.weights // new HashMap[T, HashMap[Int, Double]]()
    val varianceVectors = init.variances //new HashMap[T, HashMap[Int, Double]]()

    // Begin training loop
    println("Beginning %d rounds of training with %d instances and smoothing parameter %.2f.".format(rounds, data.size, smoothing))
    val timer = new dagger.util.Timer
    timer.start
    var classifier = new AROWClassifier[T](weightVectors, varianceVectors)
    for (r <- 1 to rounds) {
      val instances = if (shuffle) random.shuffle(data) else data
      var errors = 0.0
      for ((instance, i) <- instances.view.zipWithIndex) {
        if (printInterval > 0 && i % printInterval == 0) print("\rRound %d...instance %d...".format(r, i))

        //println(instance)
        // maxLabel refers to labels - NOT weightLabels
        val prediction = classifier.predict(instance)
        val maxLabel = prediction.maxLabel
        val maxScore = prediction.maxScore
        val icost = instance.costOf(maxLabel)
        if (instance.costOf(maxLabel) > 0) {
          errors += 1

          // correctLabels is an array of all those with cost of 0.0
          // so this line produces tuple of correct label with the lowest score from the classifier (i.e. the least good correct prediction)
          val (minCorrectLabel, minCorrectScore) = instance.correctLabels.map(l => (l, prediction.label2score(l))).toArray.sortBy(_._2).head
          val zVectorPredicted = new HashMap[Int, Double]()
          val zVectorMinCorrect = new HashMap[Int, Double]()

          // 
          val labelList = instance.labels
          val iMaxLabel = labelList.indexOf(maxLabel)

          val maxWeightLabel = instance.weightLabels(iMaxLabel)
  //        if (maxLabel != maxWeightLabel) println(maxLabel + " using weights for " + maxWeightLabel)
          val iMinCorrectLabel = labelList.indexOf(minCorrectLabel)
          val minCorrectWeightLabel = instance.weightLabels(iMinCorrectLabel)
  //        if (minCorrectLabel != minCorrectWeightLabel) println(minCorrectLabel + " using weights for " + minCorrectWeightLabel)
          for (feat <- instance.featureVector(iMaxLabel).keys) {
            //AV: The if is not needed here, you do it with getOrElse right?
            if (varianceVectors.contains(maxWeightLabel)) {
              zVectorPredicted(feat) = instance.featureVector(iMaxLabel)(feat) * varianceVectors(maxWeightLabel).getOrElse(feat, 1.0)
            } else {
              zVectorPredicted(feat) = instance.featureVector(iMaxLabel)(feat)
            }
          }
          for (feat <- instance.featureVector(iMinCorrectLabel).keys) {
            //AV: The if is not needed here, you do it with getOrElse right?
            if (varianceVectors.contains(minCorrectWeightLabel)) {
              zVectorMinCorrect(feat) = instance.featureVector(iMinCorrectLabel)(feat) * varianceVectors(minCorrectWeightLabel).getOrElse(feat, 1.0)
            } else {
              zVectorMinCorrect(feat) = instance.featureVector(iMinCorrectLabel)(feat)
            }
          }

          val preDot = dotMap(instance.featureVector(iMaxLabel), zVectorPredicted)
          val minDot = dotMap(instance.featureVector(iMinCorrectLabel), zVectorMinCorrect)
          val confidence = preDot + minDot

          val beta = 1.0 / (confidence + smoothing)
          val loss = maxScore - minCorrectScore + math.sqrt(icost)
          val alpha = loss * beta

          if (verbose) {
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

          add(weightVectors(maxWeightLabel), zVectorPredicted, -1.0 * alpha)
          add(weightVectors(minCorrectWeightLabel), zVectorMinCorrect, alpha)

          for (feat <- instance.featureVector(iMaxLabel).keys) {
            // AV: you can save yourself this if by initializing them in the beginning
            if (!varianceVectors.contains(maxWeightLabel)) varianceVectors(maxWeightLabel) = new HashMap[Int, Double]()
            varianceVectors(maxWeightLabel)(feat) = varianceVectors(maxWeightLabel).getOrElse(feat, 1.0) - beta * math.pow(zVectorPredicted(feat), 2)
          }
          for (feat <- instance.featureVector(iMinCorrectLabel).keys) {
            // AV: you can save yourself this if by initializing them in the beginning
            if (!varianceVectors.contains(minCorrectWeightLabel)) varianceVectors(minCorrectWeightLabel) = new HashMap[Int, Double]()
            varianceVectors(minCorrectWeightLabel)(feat) = varianceVectors(minCorrectWeightLabel).getOrElse(feat, 1.0) - beta * math.pow(zVectorMinCorrect(feat), 2)
          }
        }
      }
      classifier = new AROWClassifier[T](weightVectors, varianceVectors)
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
    timer.stop()
    println("Completed in %s.".format(timer.toString))

    // Return final classifier
    new AROWClassifier[T](weightVectors, varianceVectors)
  }

  def add(v1: HashMap[Int, Double], v2: HashMap[Int, Double], damp: Double = 1.0) = {
    for ((key, value) <- v2) v1(key) = v1.getOrElse(key, 0.0) + value * damp
  }

  def dotMap(v1: collection.Map[Int, Double], v2: collection.Map[Int, Double]): Double = {
    v1.foldLeft(0.0) { case (sum, (f, v)) => sum + v * v2.getOrElse(f, 0.0) }
  }

  // Remove rare features
  def removeRareFeatures[T](data: Iterable[Instance[T]], count: Int = 0): Iterable[Instance[T]] = {
    if (count == 0) return data
    val fcounts = new collection.mutable.HashMap[Int, Double].withDefaultValue(0.0)
    for (d <- data; m <- d.featureVector; f <- m) fcounts(f._1) = fcounts(f._1) + f._2
    val rareFeats = fcounts.collect { case (k, v) if v > count => k }.toSet
    val out = data.map(d => d.copy(feats = ((0 until d.feats.size).toList map
      (i => d.featureVector(i).filter { case (k, v) => rareFeats.contains(k) }))))
    out
  }

  def tuneSmoothingParameter[T: ClassTag](data: Iterable[Instance[T]], labels: Array[T], weightLabels: Array[T], init: AROWClassifier[T], random: Random): Double = {
    val (rtrain, rdev) = random.shuffle(data).partition(x => random.nextDouble() < 0.9)
    // Find smoothing with lowest aggregate cost in parameter sweep
    // Should be via a minBy but current implementation 2.10 is bad
    val best = (-3 to 3).map(math.pow(10, _)).map { s =>
      val classifier = trainFromClassifier(rtrain, rounds = 10, smoothing = s, printInterval = 0, init = init, random = random)
      val cost = rdev.map(d => d.costOf(classifier.predict(d).maxLabels.head)).foldLeft(0.0)(_ + _)
      println("Cost for smoothing parameter = %.4f is %.2f\n".format(s, cost))
      (s, cost)
    }.sortBy(_._2).head._1
    println("Optimal smoothing parameter setting is %.4f".format(best))
    best
  }

}

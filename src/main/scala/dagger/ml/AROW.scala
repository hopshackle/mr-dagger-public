package dagger.ml

import java.io.{ File, FileWriter }

import collection.mutable.HashMap
// import java.util.HashMap
import scala.reflect.ClassTag
import scala.util.Random
import dagger.core._
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

  import AROWClassifier.{ getWeights, setWeights, weightsExist }

  def predict(instance: Instance[T]): Prediction[T] = {
    //  println("instance is null ?" + instance == null)
    //  println("size:" + instance.labels.size)

    val scores = (instance.labels zip instance.featureVector) map {
      case (label, feats) =>
        if (!weightsExist(weights, label)) setWeights(label, weights, new HashMap[Int, Double])
        label -> dotMap(feats, getWeights(weights, label))
    }
    Prediction[T](label2score = scores.toMap)
  }

  //
  //    val scores: Map[T, Double] = if (weights.isEmpty) {
  //      // Actually, this if shouldn't be needed: all weights should be initialized to 0 so the dotMap always works
  //      instance.labels.map { label => label -> 0.0 }.toMap
  //    }
  //    else {
  //      instance.labels.map { label =>
  //        label -> dotMap(instance.featureVector, weights(label))
  //      }.toMap
  //    }

  def weightOf(a: T, p: Int): Double = getWeights(weights, a).getOrElse(p, 0.0)

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

  def weightsExist[T: ClassTag](w: HashMap[T, HashMap[Int, Double]], l: T): Boolean = {
    val masterLabel: T = if (l.isInstanceOf[MasterLabel]) ((l.asInstanceOf[MasterLabel]).getMasterLabel).asInstanceOf[T] else l
    w.contains(masterLabel)
  }
  def getWeights[T: ClassTag](w: HashMap[T, HashMap[Int, Double]], l: T): HashMap[Int, Double] = {
    val masterLabel: T = if (l.isInstanceOf[MasterLabel]) ((l.asInstanceOf[MasterLabel]).getMasterLabel).asInstanceOf[T] else l
    w(masterLabel)
  }
  def setWeights[T: ClassTag](l: T, w: HashMap[T, HashMap[Int, Double]], n: HashMap[Int, Double]): Unit = {
    val masterLabel: T = if (l.isInstanceOf[MasterLabel]) ((l.asInstanceOf[MasterLabel]).getMasterLabel).asInstanceOf[T] else l
    w(masterLabel) = n
  }

  def empty[T: ClassTag](labels: Array[T] = Array()): AROWClassifier[T] = {
    val weights = new HashMap[T, HashMap[Int, Double]]
    val variances = new HashMap[T, HashMap[Int, Double]]
    for (l <- labels) {
      if (!weightsExist(weights, l)) weights(l) = new HashMap[Int, Double]
      if (!weightsExist(variances, l)) variances(l) = new HashMap[Int, Double]
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
      if (!weightsExist(weights, label)) weights(label) = new HashMap[Int, Double]
      weights(label)(f) = w
    }
    AROWClassifier(weights, variances)
  }

}

object AROW {

  import AROWClassifier.{ weightsExist, getWeights, setWeights }

  def train[T: ClassTag](data: Iterable[Instance[T]], labels: Array[T], weightLabels: Array[T], options: AROWOptions, init: Option[AROWClassifier[T]] = None): AROWClassifier[T] = {
    val pruned = removeRareFeatures(data, options.RARE_FEATURE_COUNT)
    val random = new Random(options.RANDOM_SEED)
    val model = init match {
      case Some(arow) => arow
      case None => AROWClassifier.empty[T](labels = labels)
    }
    val smoothing = if (options.TUNE_REGULARIZER) {
      tuneSmoothingParameter(data = pruned, labels = labels, init = model, random)
    } else {
      options.SMOOTHING
    }
    trainFromClassifier(
      data = pruned,
      labels,
      weightLabels,
      rounds = options.TRAIN_ITERATIONS,
      shuffle = options.SHUFFLE,
      averaging = options.AVERAGING,
      smoothing = smoothing,
      init = model,
      random = random)
  }

  private def trainFromClassifier[T: ClassTag](data: Iterable[Instance[T]], labels: Array[T], weightLabels: Array[T], rounds: Int, smoothing: Double = 1.0, shuffle: Boolean = true,
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
          val iMinCorrectLabel = labelList.indexOf(minCorrectLabel)
          for (feat <- instance.featureVector(iMaxLabel).keys) {
            //AV: The if is not needed here, you do it with getOrElse right?
            if (weightsExist(varianceVectors, maxLabel)) {
              zVectorPredicted(feat) = instance.featureVector(iMaxLabel)(feat) * getWeights(varianceVectors, maxLabel).getOrElse(feat, 1.0)
            } else {
              zVectorPredicted(feat) = instance.featureVector(iMaxLabel)(feat)
            }
          }
          for (feat <- instance.featureVector(iMinCorrectLabel).keys) {
            //AV: The if is not needed here, you do it with getOrElse right?
            if (weightsExist(varianceVectors, minCorrectLabel)) {
              zVectorMinCorrect(feat) = instance.featureVector(iMinCorrectLabel)(feat) * getWeights(varianceVectors, minCorrectLabel).getOrElse(feat, 1.0)
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

          add(getWeights(weightVectors, maxLabel), zVectorPredicted, -1.0 * alpha)
          add(getWeights(weightVectors, minCorrectLabel), zVectorMinCorrect, alpha)

          for (feat <- instance.featureVector(iMaxLabel).keys) {
            // AV: you can save yourself this if by initializing them in the beginning
            val newMap = if (!weightsExist(varianceVectors, maxLabel)) new HashMap[Int, Double]()
            else getWeights(varianceVectors, maxLabel)
            newMap(feat) = newMap.getOrElse(feat, 1.0) - beta * math.pow(zVectorPredicted(feat), 2)
            setWeights(maxLabel, varianceVectors, newMap)
          }
          for (feat <- instance.featureVector(iMinCorrectLabel).keys) {
            // AV: you can save yourself this if by initializing them in the beginning
            val newMap = if (!weightsExist(varianceVectors, minCorrectLabel)) new HashMap[Int, Double]()
            else getWeights(varianceVectors, minCorrectLabel)
            newMap(feat) = newMap.getOrElse(feat, 1.0) - beta * math.pow(zVectorMinCorrect(feat), 2)
            setWeights(minCorrectLabel, varianceVectors, newMap)
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

  def tuneSmoothingParameter[T: ClassTag](data: Iterable[Instance[T]], labels: Array[T], init: AROWClassifier[T], random: Random): Double = {
    val (rtrain, rdev) = random.shuffle(data).partition(x => random.nextDouble() < 0.9)
    // Find smoothing with lowest aggregate cost in parameter sweep
    // Should be via a minBy but current implementation 2.10 is bad
    val best = (-3 to 3).map(math.pow(10, _)).map { s =>
      val classifier = trainFromClassifier(rtrain, labels, rounds = 10, smoothing = s, printInterval = 0, init = init, random = random)
      val cost = rdev.map(d => d.costOf(classifier.predict(d).maxLabels.head)).foldLeft(0.0)(_ + _)
      println("Cost for smoothing parameter = %.4f is %.2f\n".format(s, cost))
      (s, cost)
    }.sortBy(_._2).head._1
    println("Optimal smoothing parameter setting is %.4f".format(best))
    best
  }

}

package dagger.ml

import java.io.{ File, FileWriter }

import collection.mutable.HashMap
import scala.reflect.ClassTag
import scala.util.Random
import scala.language.postfixOps

/**
 * Created with IntelliJ IDEA.
 * User: narad
 * Date: 5/14/14
 * Time: 12:13 PM
 */
case class AROWClassifier[T: ClassTag](weights: HashMap[T, HashMap[Int, Float]] = new HashMap[T, HashMap[Int, Float]](),
  variances: HashMap[T, HashMap[Int, Float]] = new HashMap[T, HashMap[Int, Float]](),
  averagingCounter: Int = 1, cachedWeights: HashMap[T, HashMap[Int, Float]] = new HashMap[T, HashMap[Int, Float]]())
  extends MultiClassClassifier[T] {

  def predict(instance: Instance[T]): Prediction[T] = {
    //  println("instance is null ?" + instance == null)
    //  println("size:" + instance.labels.size)
    import scala.collection.JavaConversions._
    val scores = (instance.labels, instance.weightLabels, instance.feats).zipped map {
      case (label, weightLabel, feats) =>
        val pruned = Instance.pruneRareFeatures(feats)
        if (!weights.contains(weightLabel)) {
          weights(weightLabel) = new HashMap[Int, Float]
          cachedWeights(weightLabel) = new HashMap[Int, Float]
        }
        label -> dotMap(pruned, weights(weightLabel))
    }
    Prediction[T](label2score = scores.toMap)
  }

  def weightOf(a: T, p: Int): Float = weights.getOrElse(a, HashMap[Int, Float]()).getOrElse(p, 0.0f)

  def writeToFile(filename: String, actionToString: T => String) = {
    val file = new File(filename)
    if (!file.getParentFile.exists()) file.getParentFile.mkdirs()
    val out = new FileWriter(filename)
    for (label <- weights.keys; (f, w) <- weights(label)) {
      if (actionToString != null) out.write(actionToString(label) + "\t" + f + "\t" + w + "\n")
    }
    out.close()
  }

  override def applyAveraging: MultiClassClassifier[T] = {
    val newWeights = weights map { case (action, weights) => (action -> (weights map { case (k, v) => (k, v - cachedWeights(action)(k) / averagingCounter.toFloat) })) }
    println("Averaged using averagingCount of " + averagingCounter)
    this.copy(weights = newWeights)
  }
}

object AROWClassifier {

  def empty[T: ClassTag](weightLabels: Array[T]): AROWClassifier[T] = {
    val weights = new HashMap[T, HashMap[Int, Float]]
    val cachedWeights = new HashMap[T, HashMap[Int, Float]]
    val variances = new HashMap[T, HashMap[Int, Float]]
    for (l <- weightLabels) {
      if (!weights.contains(l)) weights(l) = new HashMap[Int, Float]
      if (!variances.contains(l)) variances(l) = new HashMap[Int, Float]
      if (!cachedWeights.contains(l)) cachedWeights(l) = new HashMap[Int, Float]
    }
    AROWClassifier(weights, variances, cachedWeights = cachedWeights)
  }

  def fromFile[T: ClassTag](filename: String, actionMap: (String => T)): AROWClassifier[T] = {
    val weights = new HashMap[T, HashMap[Int, Float]]
    val cachedWeights = new HashMap[T, HashMap[Int, Float]]
    val variances = new HashMap[T, HashMap[Int, Float]]
    val lines = io.Source.fromFile(filename).getLines
    lines.foreach { line =>
      val cols = line.split("\t")
      val label = actionMap(cols(0))
      val f = cols(1).toInt
      val w = cols(2).toFloat
      if (!weights.contains(label)) weights(label) = new HashMap[Int, Float]
      if (!cachedWeights.contains(label)) cachedWeights(label) = new HashMap[Int, Float]
      weights(label)(f) = w
    }
    AROWClassifier(weights, variances, cachedWeights = cachedWeights)
  }
}

object AROW {

  def train[T: ClassTag](data: Iterable[Instance[T]], weightLabels: Array[T], options: AROWOptions, init: Option[AROWClassifier[T]] = None): AROWClassifier[T] = {
    val rareFeatures = removeRareFeatures(data, options.RARE_FEATURE_COUNT)
    val random = new Random(options.RANDOM_SEED)
    val model = init match {
      case Some(arow) => arow
      case None => AROWClassifier.empty[T](weightLabels = weightLabels)
    }
    val smoothing = if (options.TUNE_REGULARIZER) {
      tuneSmoothingParameter(data, rareFeatures, weightLabels = weightLabels, init = model, random)
    } else {
      options.SMOOTHING
    }
    trainFromClassifier(
      data = data,
      rareFeats = rareFeatures,
      options = options,
      init = model,
      random = random)
  }

  private def trainFromClassifier[T: ClassTag](data: Iterable[Instance[T]], rareFeats: Set[Int], options: AROWOptions, init: AROWClassifier[T], random: Random): AROWClassifier[T] = {
    // Begin training loop
    val rounds = options.TRAIN_ITERATIONS
    val verbose = options.VERBOSE
    val shuffle = options.SHUFFLE
    val printInterval = options.AROW_PRINT_INTERVAL
    println("Beginning %d rounds of training with smoothing parameter %.2f.".format(rounds, options.SMOOTHING))
    val timer = new dagger.util.Timer
    timer.start
    var classifier = init
    val errors = new Array[Double](rounds)
    Instance.setRareFeatures(rareFeats)
    // hack. should be tided up. The problem was that I want to keep all the features in the instance, because the instance also
    // holds state information about number of errors made (for alpha bound), and the weight updates made historically (for averaging)
    // So the previous attempt to create a copy of the instance with pruned data failed to track this...so in validating those two
    // areas of functionality. Now the pruning occurs at the point of calculation/prediction in AROW.predict...but I store the rareFeature
    // list in the Instance companion Object so that this is clearer to someone reading that code.
    for (r <- 1 to rounds) {
      if (verbose) println("Starting round " + r)
      val instances = if (options.SHUFFLE) random.shuffle(data) else data
      for ((instance, i) <- instances.toIterator.zipWithIndex) {
        if (options.AROW_PRINT_INTERVAL > 0 && i % options.AROW_PRINT_INTERVAL == 0) print("\rRound %d...instance %d...".format(r, i))
        if (instance.getErrorCount < options.INSTANCE_ERROR_MAX) {
          if (verbose) println("Instance " + i)
          val (misClassification, newClassifier) = innerLoop(i, r, instance, options, classifier, random)
          classifier = newClassifier
          if (misClassification) errors(r - 1) += 1
        }
      }
      if (verbose) println("Training error in round %d : %1.2f".format(r, (100 * errors(r - 1) / data.size)))
    }

    // Compute final training error
    var finalErrors = 0.0
    var finalCost = 0.0
    var totalInstances = 0
    for (instance <- data) {
      val prediction = classifier.predict(instance)
      val maxLabel = prediction.randomMaxLabel(random)
      val maxCost = instance.costOf(maxLabel)
      if (maxCost > 0) {
        finalErrors += 1
        finalCost += maxCost
      }
      totalInstances += 1
    }
    println("Final training error rate (%1.0f / %d) = %1.3f".format(finalErrors, totalInstances, 100 * finalErrors / totalInstances))
    println("Final training cost = %1.3f".format(finalErrors))
    timer.stop()
    println("Completed in %s.".format(timer.toString))

    // Return final classifier
    classifier
  }

  def innerLoop[T: ClassTag](i: Int, r: Int, instance: Instance[T], options: AROWOptions, classifier: AROWClassifier[T], random: Random): (Boolean, AROWClassifier[T]) = {
    //println(instance)
    // maxLabel refers to labels - NOT weightLabels
    val weightVectors = classifier.weights
    val cachedWeightVectors = classifier.cachedWeights
    val varianceVectors = classifier.variances
    val prediction = classifier.predict(instance)
    val maxLabel = prediction.maxLabel
    val maxScore = prediction.maxScore
    val icost = instance.costOf(maxLabel)
    //       if (verbose) println(f"Prediction ${prediction}, maxLabel ${maxLabel}, maxScore ${maxScore}%.2f, iCost ${icost}%.2f")
    val error = if (icost > 0.0) true else false
    if (icost > 0.0) {
      instance.errorIncrement

      // correctLabels is an array of all those with cost of 0.0
      // so this line produces tuple of correct label with the lowest score from the classifier (i.e. the least good correct prediction)
      val temp = instance.correctLabels.map(l => (l, prediction.label2score(l))).toArray.sortBy(_._2)
      if (temp.isEmpty) println("No Correct Labels found for: \n" + instance)
      val (minCorrectLabel, minCorrectScore) = temp.head
      val zVectorPredicted = new HashMap[Int, Float]()
      val zVectorMinCorrect = new HashMap[Int, Float]()

      // 
      val labelList = instance.labels
      val iMaxLabel = labelList.indexOf(maxLabel)

      val maxWeightLabel = instance.weightLabels(iMaxLabel)
      //        if (maxLabel != maxWeightLabel) println(maxLabel + " using weights for " + maxWeightLabel)
      val iMinCorrectLabel = labelList.indexOf(minCorrectLabel)
      val minCorrectWeightLabel = instance.weightLabels(iMinCorrectLabel)
      //        if (minCorrectLabel != minCorrectWeightLabel) println(minCorrectLabel + " using weights for " + minCorrectWeightLabel)
      for (feat <- (instance.featureVector(iMaxLabel).keySet diff Instance.rareFeatures)) {
        if (varianceVectors.contains(maxWeightLabel))
          zVectorPredicted(feat) = instance.featureVector(iMaxLabel)(feat) * varianceVectors(maxWeightLabel).getOrElse(feat, 1.0f)
        else
          zVectorPredicted(feat) = instance.featureVector(iMaxLabel)(feat)
      }
      for (feat <- (instance.featureVector(iMinCorrectLabel).keySet diff Instance.rareFeatures)) {
        if (varianceVectors.contains(minCorrectWeightLabel))
          zVectorMinCorrect(feat) = instance.featureVector(iMinCorrectLabel)(feat) * varianceVectors(minCorrectWeightLabel).getOrElse(feat, 1.0f)
        else
          zVectorMinCorrect(feat) = instance.featureVector(iMinCorrectLabel)(feat)
      }

      val preDot = dotMap(instance.featureVector(iMaxLabel), zVectorPredicted)
      val minDot = dotMap(instance.featureVector(iMinCorrectLabel), zVectorMinCorrect)
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

      add(weightVectors(maxWeightLabel), zVectorPredicted, -1.0f * alpha)
      add(weightVectors(minCorrectWeightLabel), zVectorMinCorrect, alpha)
      if (options.AVERAGING) {
        add(cachedWeightVectors(maxWeightLabel), zVectorPredicted, -1.0f * alpha * classifier.averagingCounter)
        add(cachedWeightVectors(minCorrectWeightLabel), zVectorMinCorrect, alpha * classifier.averagingCounter)
      }

      for (feat <- instance.featureVector(iMaxLabel).keySet diff Instance.rareFeatures) {
        // AV: you can save yourself this if by initializing them in the beginning
        if (!varianceVectors.contains(maxWeightLabel)) varianceVectors(maxWeightLabel) = new HashMap[Int, Float]
        varianceVectors(maxWeightLabel)(feat) = varianceVectors(maxWeightLabel).getOrElse(feat, 1.0f) - beta * math.pow(zVectorPredicted(feat), 2).toFloat
      }
      for (feat <- instance.featureVector(iMinCorrectLabel).keySet diff Instance.rareFeatures) {
        // AV: you can save yourself this if by initializing them in the beginning
        if (!varianceVectors.contains(minCorrectWeightLabel)) varianceVectors(minCorrectWeightLabel) = new HashMap[Int, Float]
        varianceVectors(minCorrectWeightLabel)(feat) = (varianceVectors(minCorrectWeightLabel).getOrElse(feat, 1.0f) - beta * math.pow(zVectorMinCorrect(feat), 2)).toFloat
      }
    }
    (error, new AROWClassifier[T](weightVectors, varianceVectors, classifier.averagingCounter + 1, cachedWeightVectors))
  }

  def add(v1: HashMap[Int, Float], v2: HashMap[Int, Float], damp: Float = 1.0f) = {
    for ((key, value) <- v2) v1(key) = v1.getOrElse(key, 0.0f) + value * damp
  }

  def dotMap(v1: collection.Map[Int, Float], v2: collection.Map[Int, Float]): Float = {
    v1.foldLeft(0.0f) { case (sum, (f, v)) => sum + v * v2.getOrElse(f, 0.0f) }
  }

  def dotMap(v1: gnu.trove.map.hash.THashMap[Int, Float], v2: collection.Map[Int, Float]): Float = {
    val scalaMap = Instance.troveMapToScala(v1)
    dotMap(scalaMap, v2)
  }

  def removeRareFeatures[T](data: Iterable[Instance[T]], count: Int = 0): Set[Int] = {
    println("Rare Feature Count = " + count)
    if (count == 0) return Set()
    println("Removing Rare Features")
    val fcounts = new collection.mutable.HashMap[Int, Int].withDefaultValue(0)
    // To avoid multi-counting, we take the distinct features from each instance (which has multiple featureVectors)
    val reducedFeatures = for {
      d <- data
      keys = d.featureVector.flatten.map { _._1 }.distinct
    } yield keys

    for (d <- reducedFeatures; f <- d) fcounts(f) = fcounts(f) + 1

    val rareFeats = fcounts filter (_._2 <= count) map (_._1) toSet

    println(s"A Total of ${rareFeats.size} features removed, with ${fcounts.size - rareFeats.size} remaining.")
    rareFeats
  }

  def tuneSmoothingParameter[T: ClassTag](data: Iterable[Instance[T]], rareFeats: Set[Int], weightLabels: Array[T], init: AROWClassifier[T], random: Random): Double = {
    val (rtrain, rdev) = random.shuffle(data).partition(x => random.nextDouble() < 0.9)
    // Find smoothing with lowest aggregate cost in parameter sweep
    // Should be via a minBy but current implementation 2.10 is bad
    val best = (-3 to 3).map(math.pow(10, _)).map { s =>
      val opt = new AROWOptions(Array[String]("--arow.smoothing", s.toString))
      opt.TRAIN_ITERATIONS = 10
      val classifier = trainFromClassifier(rtrain, rareFeats, opt, init = init, random = random)
      val cost = rdev.map(d => d.costOf(classifier.predict(d).maxLabels.head)).foldLeft(0.0)(_ + _)
      println("Cost for smoothing parameter = %.4f is %.2f\n".format(s, cost))
      (s, cost)
    }.sortBy(_._2).head._1
    println("Optimal smoothing parameter setting is %.4f".format(best))
    best
  }

}

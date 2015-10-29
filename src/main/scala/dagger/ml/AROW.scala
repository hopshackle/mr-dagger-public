package dagger.ml

import java.io.{ File, FileWriter }

import collection.mutable.HashMap
import scala.reflect.ClassTag
import scala.util.Random

/**
 * Created with IntelliJ IDEA.
 * User: narad
 * Date: 5/14/14
 * Time: 12:13 PM
 */
case class AROWClassifier[T: ClassTag](weights: HashMap[T, HashMap[Int, Float]] = new HashMap[T, HashMap[Int, Float]](),
  variances: HashMap[T, HashMap[Int, Float]] = new HashMap[T, HashMap[Int, Float]]())
  extends MultiClassClassifier[T] {

  def predict(instance: Instance[T]): Prediction[T] = {
    //  println("instance is null ?" + instance == null)
    //  println("size:" + instance.labels.size)

    val scores = (instance.labels, instance.weightLabels, instance.featureVector).zipped map {
      case (label, weightLabel, feats) =>
        if (!weights.contains(weightLabel)) weights(weightLabel) = new HashMap[Int, Float]
        label -> dotMap(feats, weights(weightLabel))
    }
    Prediction[T](label2score = scores.toMap)
  }

  def weightOf(a: T, p: Int): Float = weights.getOrElse(a, HashMap[Int, Float]()).getOrElse(p, 0.0f)

  def writeToFile(filename: String, actionToString: T => String) = {
    val file = new File(filename)
    if (!file.getParentFile.exists()) file.getParentFile.mkdirs()
    val out = new FileWriter(filename)
    for (label <- weights.keys; (f, w) <- weights(label)) {
      out.write(actionToString(label) + "\t" + f + "\t" + w + "\n")
    }
    out.close()
  }
}

object AROWClassifier {

  def empty[T: ClassTag](weightLabels: Array[T]): AROWClassifier[T] = {
    val weights = new HashMap[T, HashMap[Int, Float]]
    val variances = new HashMap[T, HashMap[Int, Float]]
    for (l <- weightLabels) {
      if (!weights.contains(l)) weights(l) = new HashMap[Int, Float]
      if (!variances.contains(l)) variances(l) = new HashMap[Int, Float]
    }
    AROWClassifier(weights, variances)
  }

  def fromFile[T: ClassTag](filename: String, actionMap: (String => T)): AROWClassifier[T] = {
    val weights = new HashMap[T, HashMap[Int, Float]]
    val variances = new HashMap[T, HashMap[Int, Float]]
    val lines = io.Source.fromFile(filename).getLines
    lines.foreach { line =>
      val cols = line.split("\t")
      val label = actionMap(cols(0))
      val f = cols(1).toInt
      val w = cols(2).toFloat
      if (!weights.contains(label)) weights(label) = new HashMap[Int, Float]
      weights(label)(f) = w
    }
    AROWClassifier(weights, variances)
  }
}

object AROW {

  def train[T: ClassTag](data: Iterable[Instance[T]], weightLabels: Array[T], options: AROWOptions, init: Option[AROWClassifier[T]] = None): AROWClassifier[T] = {
    val pruned = removeRareFeatures(data, options.RARE_FEATURE_COUNT)
    val random = new Random(options.RANDOM_SEED)
    val model = init match {
      case Some(arow) => arow
      case None => AROWClassifier.empty[T](weightLabels = weightLabels)
    }
    val smoothing = if (options.TUNE_REGULARIZER) {
      tuneSmoothingParameter(data = pruned, weightLabels = weightLabels, init = model, random)
    } else {
      options.SMOOTHING
    }
    trainFromClassifier(
      data = pruned,
      options = options,
      init = model,
      random = random)
  }

  private def trainFromClassifier[T: ClassTag](data: Iterable[Instance[T]], options: AROWOptions, init: AROWClassifier[T], random: Random): AROWClassifier[T] = {
    // Initialize weight and variance vectors
    val weightVectors = init.weights 
    val varianceVectors = init.variances 
    
    // Begin training loop
    val rounds = options.TRAIN_ITERATIONS
    val verbose = options.VERBOSE
    val averaging = options.AVERAGING
    val shuffle = options.SHUFFLE
    val printInterval = options.AROW_PRINT_INTERVAL
    println("Beginning %d rounds of training with %d instances and smoothing parameter %.2f.".format(rounds, data.size, options.SMOOTHING))
    val timer = new dagger.util.Timer
    timer.start
    var classifier = new AROWClassifier[T](weightVectors, varianceVectors)
    val errors = new Array[Double](rounds)
    var averagedClassifier = classifier

    for (r <- 1 to (if (options.AVERAGING) 1 else rounds)) {
      if (verbose) println("Starting round " + r)
      val instances = (if (options.SHUFFLE) random.shuffle(data) else data) filter (in => in.getErrorCount < options.INSTANCE_ERROR_MAX)
      for ((instance, i) <- instances.view.zipWithIndex) {
        if (verbose) println("Instance " + i)
        for (r2 <- 1 to (if (averaging) rounds else 1)) {
          val actualRound = math.max(r, r2)
          val (e, newClassifier) = innerLoop(i, actualRound, instance, options, classifier, random)
          classifier = newClassifier
          errors(actualRound - 1) += e
        }
        if (averaging) {
          averagedClassifier = if (i == 0) classifier else average(averagedClassifier, classifier, i)
          classifier = averagedClassifier
          if (verbose) println("Have averaged")
        }
      }
      if (verbose && !averaging) println("Training error in round %d : %1.2f".format(r, (100 * errors(r - 1) / data.size)))
      if (verbose && averaging) {
        for (loop <- 1 to rounds) println("Training error in round %d : %1.2f".format(loop, (100 * errors(loop - 1) / data.size)))
      }
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

  def innerLoop[T: ClassTag](i: Int, r: Int, instance: Instance[T], options: AROWOptions, classifier: AROWClassifier[T], random: Random): (Double, AROWClassifier[T]) = {

    var errors = 0.0
    if (options.AROW_PRINT_INTERVAL > 0 && i % options.AROW_PRINT_INTERVAL == 0) print("\rRound %d...instance %d...".format(r, i))

    //println(instance)
    // maxLabel refers to labels - NOT weightLabels
    val weightVectors = classifier.weights
    val varianceVectors = classifier.variances
    val prediction = classifier.predict(instance)
    val maxLabel = prediction.maxLabel
    val maxScore = prediction.maxScore
    val icost = instance.costOf(maxLabel)
    //       if (verbose) println(f"Prediction ${prediction}, maxLabel ${maxLabel}, maxScore ${maxScore}%.2f, iCost ${icost}%.2f")
    if (icost > 0.0) {
      errors += 1
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
      for (feat <- instance.featureVector(iMaxLabel).keys) {
        if (varianceVectors.contains(maxWeightLabel))
          zVectorPredicted(feat) = instance.featureVector(iMaxLabel)(feat) * varianceVectors(maxWeightLabel).getOrElse(feat, 1.0f)
        else
          zVectorPredicted(feat) = instance.featureVector(iMaxLabel)(feat)
      }
      for (feat <- instance.featureVector(iMinCorrectLabel).keys) {
        if (varianceVectors.contains(minCorrectWeightLabel))
          zVectorMinCorrect(feat) = instance.featureVector(iMinCorrectLabel)(feat) * varianceVectors(minCorrectWeightLabel).getOrElse(feat, 1.0f)
        else
          zVectorMinCorrect(feat) = instance.featureVector(iMinCorrectLabel)(feat)
      }
      
      // If we had a shenanigan feature with value 1.0, then this would be unique for each of maxLabel and minCorrectLabel.
      // We do not need to update any value for its weight, as by definition it will never be used again.
      // Its impact is purely on the confidence weighting, as we have an additional variance value of the default of 1.0 in each case
      // Hence the impact of the shenanigan is simply to add a constant to each of preDot and minDot
      // BUT - this is exactly the same as increasing smoothing!!!!)
      
      // OK...but if we had a shenanigan feature, then this weight would be updated, and on the next iteration through this training
      // instance this would increase the likelihood of selecting the correct answer - which would mean that the weights are less likely
      // to be updated on future iterations (also a regularisation impact). If this is the case, then I would expect shenanigans to have no impact on 
      // a single AROW iteration and a single Dagger iteration.
      

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

      for (feat <- instance.featureVector(iMaxLabel).keys) {
        // AV: you can save yourself this if by initializing them in the beginning
        if (!varianceVectors.contains(maxWeightLabel)) varianceVectors(maxWeightLabel) = new HashMap[Int, Float]
        varianceVectors(maxWeightLabel)(feat) = varianceVectors(maxWeightLabel).getOrElse(feat, 1.0f) - beta * math.pow(zVectorPredicted(feat), 2).toFloat
      }
      for (feat <- instance.featureVector(iMinCorrectLabel).keys) {
        // AV: you can save yourself this if by initializing them in the beginning
        if (!varianceVectors.contains(minCorrectWeightLabel)) varianceVectors(minCorrectWeightLabel) = new HashMap[Int, Float]
        varianceVectors(minCorrectWeightLabel)(feat) = (varianceVectors(minCorrectWeightLabel).getOrElse(feat, 1.0f) - beta * math.pow(zVectorMinCorrect(feat), 2)).toFloat
      }
    }
    (errors, new AROWClassifier[T](weightVectors, varianceVectors))
  }
  
  def average[T: ClassTag](baseline: AROWClassifier[T], newbie: AROWClassifier[T], previousCount: Int): AROWClassifier[T] = {
    // Take each weight in baseline, and add previousCount / previousCount + 1
    // Take each weight in newbie and add 1 / previousCount + 1
    // No need to worry about variance at this stage
    val baseFraction = previousCount / (previousCount + 1.0f)
    val newFraction = 1.0f / (previousCount + 1.0f)
    val baseWeights = baseline.weights map { case(label, hashmap) => (label -> (new HashMap[Int, Float] ++= (hashmap.keys map {k =>  (k -> hashmap(k) * baseFraction)})))}
    val newWeights = newbie.weights map { case(label, hashmap) => (label -> (new HashMap[Int, Float] ++= (hashmap.keys map {k => (k -> hashmap(k) * newFraction)})))}
    newWeights.keys foreach (k => if (!baseWeights.contains(k)) baseWeights(k) = new HashMap[Int, Float])
    newWeights.keys foreach (k => add(baseWeights(k), newWeights(k)))
    
    new AROWClassifier[T](baseWeights, baseline.variances)
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

  // Remove rare features
  def removeRareFeatures[T](data: Iterable[Instance[T]], count: Int = 0): Iterable[Instance[T]] = {
    println("Rare Feature Count = " + count)
    if (count == 0) return data
    println("Removing Rare Features")
    val fcounts = new collection.mutable.HashMap[Int, Double].withDefaultValue(0.0)
    // To avoid multi-counting, we take the distinct features from each instance (which has multiple featureVectors)
    val reducedFeatures = for {
      d <- data
      keys = d.featureVector map identity flatMap identity groupBy (_._1) mapValues (_.map(_._2))
      maxValueMap = keys map { case (k, v) => (k, (v map Math.abs).max) }
    } yield maxValueMap

    for (d <- reducedFeatures; f <- d) fcounts(f._1) = fcounts(f._1) + f._2

    val rareFeats = fcounts.collect { case (k, v) if v > count => k }.toSet
    println(s"A Total of ${rareFeats.size} features remaining, with ${fcounts.size - rareFeats.size} removed.")
    val out = data.map(d => d.copy(feats = ((0 until d.feats.size).toList map
      (i => d.featureVector(i).filter { case (k, v) => rareFeats.contains(k) })) map Instance.scalaMapToTrove))
    out
  }

  def tuneSmoothingParameter[T: ClassTag](data: Iterable[Instance[T]], weightLabels: Array[T], init: AROWClassifier[T], random: Random): Double = {
    val (rtrain, rdev) = random.shuffle(data).partition(x => random.nextDouble() < 0.9)
    // Find smoothing with lowest aggregate cost in parameter sweep
    // Should be via a minBy but current implementation 2.10 is bad
    val best = (-3 to 3).map(math.pow(10, _)).map { s =>
      val opt = new AROWOptions(Array[String]("--arow.smoothing", s.toString))
      opt.TRAIN_ITERATIONS = 10
      val classifier = trainFromClassifier(rtrain, opt, init = init, random = random)
      val cost = rdev.map(d => d.costOf(classifier.predict(d).maxLabels.head)).foldLeft(0.0)(_ + _)
      println("Cost for smoothing parameter = %.4f is %.2f\n".format(s, cost))
      (s, cost)
    }.sortBy(_._2).head._1
    println("Optimal smoothing parameter setting is %.4f".format(best))
    best
  }

}

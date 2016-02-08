package dagger.ml

import java.io.{ File, FileWriter }

import collection.Map
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
case class AROWClassifier[T: ClassTag](weights: HashMap[T, HashMap[Int, Float]] = HashMap[T, HashMap[Int, Float]](),
  variances: HashMap[T, HashMap[Int, Float]] = new HashMap[T, HashMap[Int, Float]](),
  averagingCounter: Int = 1, cachedWeights: HashMap[T, HashMap[Int, Float]] = HashMap[T, HashMap[Int, Float]]())
  extends MultiClassClassifier[T] {

  def predict(instance: Instance[T]): Prediction[T] = {
    val scores = (instance.labels, instance.weightLabels, (0 until instance.labels.size)).zipped map {
      case (label, weightLabel, index) =>
        val pruned = if (Instance.rareFeatures.isEmpty) instance.feats(index) else Instance.pruneRareFeatures(instance.feats(index))
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
    val updateRule: PerceptronUpdateRule[T] = options.CLASSIFIER match {
      case "PA" => new PassiveAggressiveUpdate[T]()
      case "AROW" => new AROWUpdate[T]()
      case "PERCEPTRON" => new PerceptronUpdateRule[T]()
    }

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
          val (misClassification, newClassifier) = innerLoop(i, r, instance, options, classifier, random, updateRule)
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

  def innerLoop[T: ClassTag](i: Int, r: Int, instance: Instance[T], options: AROWOptions, classifier: AROWClassifier[T], 
      random: Random, updateRule: PerceptronUpdateRule[T]): (Boolean, AROWClassifier[T]) = {
    val prediction = classifier.predict(instance)
    val maxLabel = prediction.maxLabel
    val icost = instance.costOf(maxLabel)
    val error = if (icost > 0.0) true else false
    if (error) {
      instance.errorIncrement
      // updates classifier in situ
      updateRule.update(instance, classifier, options)
    }
    (error, classifier.copy(averagingCounter = classifier.averagingCounter + 1))
  }

  def add(v1: HashMap[Int, Float], v2: Map[Int, Float], damp: Float = 1.0f) = {
    for ((key, value) <- v2) v1(key) = v1.getOrElse(key, 0.0f) + value * damp
  }
/*
 * dotMap assumes that the smaller map is the first argument, so there is less to iterate over.
 * This should usually be the feature vector, not the weights
 */
  def dotMap(v1: Map[Int, Float], v2: Map[Int, Float]): Float = {
    v1.foldLeft(0.0f) { case (sum, (f, v)) => sum + v * v2.getOrElse(f, 0.0f) }
  }

  def removeRareFeatures[T](data: Iterable[Instance[T]], count: Int = 0): Set[Int] = {
    println("Rare Feature Count = " + count)
    if (count == 0) return Set()
    println("Removing Rare Features")
    val fcounts = new HashMap[Int, Int].withDefaultValue(0)
    // To avoid multi-counting, we take the distinct features from each instance (which has multiple featureVectors)
    val reducedFeatures = for {
      d <- data
      keys = d.coreFeats.keySet ++ (d.parameterFeats flatMap {
        case (k, v) => v.keySet
      })
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

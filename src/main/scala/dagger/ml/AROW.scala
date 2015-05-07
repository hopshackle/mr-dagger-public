package coref.ml

import java.io.{File, FileWriter}

import coref.system._

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
    val scores = instance.labels.map { label =>
      if (!weights.contains(label)) weights(label) = new HashMap[Int, Double]
      label -> dotMap(instance.featureVector, weights(label))
    }.toMap
    Prediction[T](label2score = scores)
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


  def weightOf(a: T, p: Int): Double = weights(a).getOrElse(p, 0.0)

  def writeToFile(filename: String) = {
    println("Labels: " + weights.keys.mkString(", "))
    val file = new File(filename)
    if (!file.getParentFile.exists()) file.getParentFile.mkdirs()
    val out = new FileWriter(filename)
    for (label <- weights.keys; (f,w) <- weights(label)) {
        out.write(label + "\t" + f + "\t" + w + "\n")
    }
    out.close()
  }
}

object AROWClassifier {

  def empty[T: ClassTag](labels: Array[T] = Array()): AROWClassifier[T] = {
    val weights   = new HashMap[T, HashMap[Int, Double]]
    val variances = new HashMap[T, HashMap[Int, Double]]
    for (l <- labels) {
      if (!weights.contains(l)) weights(l) = new HashMap[Int, Double]
      if (!variances.contains(l)) variances(l) = new HashMap[Int, Double]
    }
    AROWClassifier(weights, variances)
  }

  def fromFile[T: ClassTag](filename: String): AROWClassifier[T] = {
    val weights = new HashMap[T, HashMap[Int, Double]]
    val variances = new HashMap[T, HashMap[Int, Double]]
    val lines = io.Source.fromFile(filename).getLines
    lines.foreach { line =>
      val cols = line.split("\t")
      val label = actionMap[T](cols(0))
      val f = cols(1).toInt
      val w = cols(2).toDouble
      if (!weights.contains(label)) weights(label) = new HashMap[Int,Double]
      weights(label)(f) = w
    }
    AROWClassifier(weights, variances)
  }

  def actionMap[T](str: String): T = {
    val matched = str match {
      case "MentionAction" => MentionAction
      case "NoMentionAction" => NoMentionAction
      case "CorefLinkAction" => CorefLinkAction
      case "CorefNoLinkAction" => CorefNoLinkAction
      case "CorefNewClusterAction" => CorefNewClusterAction
    }
    matched.asInstanceOf[T]
  }
}


object AROW {

  def train[T: ClassTag](data: Iterable[Instance[T]], labels: Array[T], options: AROWOptions, init: Option[AROWClassifier[T]] = None): AROWClassifier[T] = {
    val pruned = removeRareFeatures(data, options.RARE_FEATURE_COUNT)
    val random = new Random(options.RANDOM_SEED)
    val model = init match {
      case Some(arow) => arow
      case None => AROWClassifier.empty[T](labels = labels)
    }
    val smoothing = if (options.TUNE_REGULARIZER)  {
      tuneSmoothingParameter(data = pruned, labels = labels, init = model, random)
    }
    else {
      options.SMOOTHING
    }
    trainFromClassifier(
      data = pruned,
      labels,
      rounds = options.TRAIN_ITERATIONS,
      shuffle = options.SHUFFLE,
      averaging = options.AVERAGING,
      smoothing = smoothing,
      init = model,
      random = random)
  }

  private def trainFromClassifier[T: ClassTag](data: Iterable[Instance[T]], labels: Array[T], rounds: Int, smoothing: Double = 1.0, shuffle: Boolean = true,
                                 averaging: Boolean = true, init: AROWClassifier[T], printInterval: Int = 100000, verbose: Boolean = false, random: Random): AROWClassifier[T] = {
    // Initialize weight and variance vectors
    val weightVectors   = init.weights // new HashMap[T, HashMap[Int, Double]]()
    val varianceVectors = init.variances //new HashMap[T, HashMap[Int, Double]]()

    // Begin training loop
    println("Beginning %d rounds of training with %d instances and smoothing parameter %.2f.".format(rounds, data.size, smoothing))
    val timer = new coref.util.Timer
    timer.start
    var classifier = new AROWClassifier[T](weightVectors, varianceVectors)
    for (r <- 1 to rounds) {
      val instances = if (shuffle) random.shuffle(data) else data
      var errors = 0.0
      for ((instance, i) <- instances.view.zipWithIndex) {
        if (printInterval > 0 && i % printInterval == 0) print("\rRound %d...instance %d...".format(r, i))

        val prediction = classifier.predict(instance)
        val maxLabel = prediction.maxLabel
        val maxScore = prediction.maxScore
        val icost = instance.costOf(maxLabel)
        if (instance.costOf(maxLabel) > 0) {
          errors += 1

          val (minCorrectLabel, minCorrectScore) = instance.correctLabels.map(l => (l, prediction.label2score(l))).toArray.sortBy(_._2).head
          val zVectorPredicted = new HashMap[Int, Double]()
          val zVectorMinCorrect = new HashMap[Int, Double]()

          for (feat <- instance.featureVector.keys) {
            //AV: The if is not needed here, you do it with getOrElse right?
            if (varianceVectors.contains(maxLabel)) {
              zVectorPredicted(feat) = instance.featureVector(feat) * varianceVectors(maxLabel).getOrElse(feat, 1.0)
            }
            else {
              zVectorPredicted(feat) = instance.featureVector(feat)
            }
            //AV: The if is not needed here, you do it with getOrElse right?
            if (varianceVectors.contains(minCorrectLabel)) {
              zVectorMinCorrect(feat) = instance.featureVector(feat) * varianceVectors(minCorrectLabel).getOrElse(feat, 1.0)
            }
            else {
              zVectorMinCorrect(feat) = instance.featureVector(feat)
            }
          }

          val preDot = dotMap(instance.featureVector, zVectorPredicted)
          val minDot = dotMap(instance.featureVector, zVectorMinCorrect)
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

          add(weightVectors(maxLabel), zVectorPredicted, -1.0 * alpha)
          add(weightVectors(minCorrectLabel), zVectorMinCorrect, alpha)

          for (feat <- instance.featureVector.keys) {
            // AV: you can save yourself this if by initializing them in the beginning
            if (!varianceVectors.contains(maxLabel)) varianceVectors(maxLabel) = new HashMap[Int, Double]()
            varianceVectors(maxLabel)(feat) = varianceVectors(maxLabel).getOrElse(feat, 1.0) - beta * math.pow(zVectorPredicted(feat), 2)
            // AV: you can save yourself this if by initializing them in the beginning
            if (!varianceVectors.contains(minCorrectLabel)) varianceVectors(minCorrectLabel) = new HashMap[Int, Double]()
            varianceVectors(minCorrectLabel)(feat) = varianceVectors(minCorrectLabel).getOrElse(feat, 1.0) - beta * math.pow(zVectorMinCorrect(feat), 2)
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
    for ((key,value) <- v2) v1(key) = v1.getOrElse(key,0.0) + value * damp
  }

  def dotMap(v1: collection.Map[Int, Double], v2: collection.Map[Int, Double]): Double = {
    v1.foldLeft(0.0){ case(sum, (f,v)) => sum + v * v2.getOrElse(f, 0.0)}
  }

  // Remove rare features
  def removeRareFeatures[T](data: Iterable[Instance[T]], count: Int = 0): Iterable[Instance[T]] = {
    if (count == 0) return data
    val fcounts = new collection.mutable.HashMap[Int, Double].withDefaultValue(0.0)
    for (d <- data; f <- d.featureVector) fcounts(f._1) = fcounts(f._1) + f._2
    val rareFeats = fcounts.collect { case (k, v) if v > count => k}.toSet
    data.map(d => d.copy(feats = d.featureVector.filter{ case(k,v) => rareFeats.contains(k) }))
  }

  def tuneSmoothingParameter[T: ClassTag](data: Iterable[Instance[T]], labels: Array[T], init: AROWClassifier[T], random: Random): Double = {
    val (rtrain, rdev) = random.shuffle(data).partition(x => random.nextDouble() < 0.9)
    // Find smoothing with lowest aggregate cost in parameter sweep
    // Should be via a minBy but current implementation 2.10 is bad
    val best = (-3 to 3).map(math.pow(10,_)).map { s =>
      val classifier = trainFromClassifier(rtrain, labels, rounds = 10, smoothing = s, printInterval = 0, init = init, random = random)
      val cost = rdev.map ( d => d.costOf(classifier.predict(d).maxLabels.head)).foldLeft(0.0)(_+_)
      println("Cost for smoothing parameter = %.4f is %.2f\n".format(s, cost))
      (s, cost)
    }.sortBy(_._2).head._1
    println("Optimal smoothing parameter setting is %.4f".format(best))
    best
  }

}











/*
    if (averaging) {
      new AROWClassifier[T](averageVectors, varianceVectors)
    }
    else {
 */


//      if (averaging) {
//        for (label <- labels; wf <- weightVectors(label)) {
//          averageVectors(label)(wf._1) = averageVectors(label).getOrElse(wf._1, 0.0) + (wf._2 / rounds)
//        }
//      }

//    for (label <- labels) {
//      if (init == null) {
//        weightVectors(label)  = new HashMap[Int, Double]()
//        averageVectors(label) = new HashMap[Int, Double]()
//      }
//      else {
//        weightVectors(label)   = init.weights.getOrElse(label, new HashMap[Int, Double]())
//        varianceVectors(label) = init.variances.getOrElse(label, new HashMap[Int, Double]())
//        averageVectors(label)  = new HashMap[Int, Double]()
//      }
//    }

//          var minCorrectScore = instance.correctCost
//          var minCorrectLabel = instance.correctLabels.head
//          for (label <- Array(minCorrectLabel)) {
//            val score = dotMap(instance.featureVector, weightVectors(label))
//            if (score < minCorrectScore) {
//              minCorrectScore = score
//              minCorrectLabel = label
//            }
//          }



/*
//    println(instance.labels.mkString(", "))
val prediction = new Prediction[T]()
if (weights.isEmpty) {
  for (label <- instance.labels) {
    prediction.label2score(label) = 1.0
  }
}
else {
  for (label <- instance.labels) {
    val score = dotMap(instance.featureVector, weights(label))
    prediction.label2score(label) = dotMap(instance.featureVector, weights(label))
  }
}
prediction
}
*/


//
//def save(filename: String = "model.txt") {
//println("Saving...")
//val out = new FileWriter(filename)
//for (l <- currentWeights.keys; f <- currentWeights(l).keys) {
//out.write("%s\t%s\t%f\n".format(l, f, currentWeights(l)(f)))
//}
//out.close()
//}
//

//  var currentWeights = new HashMap[T, HashMap[String, Double]]
//  var currentVariances = new HashMap[T, HashMap[String, Double]]
//
//  def predict(instance: Instance[T]) = {
//    predict(instance, currentWeights, false, false)
//  }
//
//  def predict(instance: Instance[T], weightVector: Map[T, Map[String, Double]] = currentWeights,
//              verbose: Boolean=false, probs: Boolean = false): Prediction[T] = {
////    instance.featureVector("BIAS") = 1.0     Put the Bias in during normal feature extraction
//    val prediction = new Prediction[T]()
//    if (weightVector.isEmpty) {
//      for (label <- instance.labels) {
//        prediction.label2score(label) = 1.0
//      }
//    }
//    else {
//      for (label <- instance.labels) {
//        val score = dotMap(instance.featureVector, weightVector(label))
//        prediction.label2score(label) = dotMap(instance.featureVector, weightVector(label))
//      }
//    }
//    prediction
//  }



//  def dot(v1: Array[Double], v2: Array[Double]): Double = {
//    v1.zip(v2).map(p => p._1 * p._2).foldLeft(0.0)(_+_)
//  }
//
//  def batchPredict() {}
//
//  def probGeneration() {}
//
//  def trainOpt() {}
//
//  def save(filename: String = "model.txt") {
//    println("Saving...")
//    val out = new FileWriter(filename)
//    for (l <- currentWeights.keys; f <- currentWeights(l).keys) {
//      out.write("%s\t%s\t%f\n".format(l, f, currentWeights(l)(f)))
//    }
//    out.close()
//  }
//
//  def load() {}



//  object AROW
/*
 def main(args: Array[String]) {
   val classifier = new AROW[String]
//    val data = new LibSVMReader(args(0)).toArray
   val data = Random.shuffle(new LibSVMReader(args(0))).toArray
   val train = data.slice(0,15000)
   val test: Array[Instance[String]] = data.slice(15000, data.size)
   println("Training Set contains %d instances.".format(train.size))
   println("Test Set contains %d instances.".format(test.size))
   val labels = Array("+1", "-1")
   classifier.train(train, labels, rounds = 10, smooth = 1.0)
   var correct = 0.0
   for (d <- test) {
     val label = classifier.predict(d).maxLabels.head
     if (d.correctLabels.contains(label)) correct += 1
   }
   println("AROW Accuracy = %.3f".format(correct / test.size))
 }

 def generateEvenOddsData(size: Int): Array[Instance[String]] = {
   val data = new ArrayBuffer[Instance[String]]
   val labels = Array("+1", "-1")
   for (i <- 1 to size) {
     if (Random.nextInt() > 0) {
       val feats = new HashMap[String, Double]()
       Array.fill(100)(Random.nextInt).filter(_ % 2 == 0).distinct.foreach(f => feats(f.toString) = 1.0)
       val costs = Array(0.0, 1.0)
       val instance = new Instance[String](feats, labels, costs)
       data += instance
     }
     else {
       val feats = new HashMap[String, Double]()
       Array.fill(100)(Random.nextInt).filter(_ % 2 == 1).distinct.foreach(f => feats(f.toString) = 1.0)
       val costs = Array(1.0, 0.0)
       val instance = new Instance[String](feats, labels, costs)
       data += instance
     }
   }
   data.toArray
 }
}
*/
















/*


  def train(data: Array[(ParserConfiguration, TransitionAction)]): Array[Double] = {
    val dict = new HashIndex(10000000)
    val actions: Array[TransitionAction] = Array(new ShiftAction, new LeftArcAction, new RightArcAction)
    for (i <- 1 to 100) {
      data.foreach { case(conf, action) =>
        val feats = features(conf)
        actions.head.toString()
        val cfeats = actions.map{a => feats.map{f => dict.index(a.toString() + f)}}
        //       update(cfeats, action)
        //.map(dict.index(_))

      }
    }
    Array()
  }

      prediction.label2score(label) = score
      if (score > prediction.score) {
        prediction.score = score
        prediction.label = label
      }

      */
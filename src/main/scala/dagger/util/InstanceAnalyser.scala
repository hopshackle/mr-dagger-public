package dagger.util
import dagger.ml._
import java.io._
import scala.reflect.ClassTag

object InstanceAnalyser {

  def basicErrorMetrics[T](instances: Iterator[Instance[T]]): Map[Int, Int] = {
    val errorCount = instances map (_.getErrorCount)
    errorCount.toList groupBy identity mapValues (_.size)
  }

  def errorsByActionType[T](instances: Iterator[Instance[T]], nameFunction: (T => String)): Map[String, (Int, Seq[Int])] = {
    // key is the basic action name, tuple._1 is the total number of instances for which this was the best action
    // tuple._2 then provide the error counts (i.e. tuple._2(i) is the number of instances with errorCount == 1)

    def mapToSeq(input: Map[Int, Int]): Seq[Int] = {
      val largestKey = input.keys.max
      val output = new Array[Int](largestKey + 1)
      input foreach {
        case (k, v) => output(k) = input(k)
      }
      output.toSeq
    }

    val labelsAndErrorCounts = instances map (i => (nameFunction(i.correctLabels.head), i.getErrorCount))
    val temp = labelsAndErrorCounts.toList groupBy (_._1)
    val total = temp mapValues (_.size)
    val detail = temp mapValues (v => v map (_._2) groupBy identity mapValues (_.size))
    val detail2 = detail mapValues mapToSeq

    total map { case (k, t) => (k, (t, detail2.getOrElse(k, Seq(0)))) }
  }

  def featureDescription[T](instances: Iterator[Instance[T]], threshold: Int, featureName: (Int => String), fileName: String): Unit = {
    val instancesToLog = instances filter (_.getErrorCount >= threshold)
    val output = new FileWriter(fileName)
    instancesToLog foreach {
      i =>
        val correctLabel = i.correctLabels.head
        val labelIndex = i.labels indexOf correctLabel
        val features = i.featureVector(labelIndex)
        output.write("Correct Action: " + correctLabel + "\n")
        output.write("Errors Made   : " + i.getErrorCount + "\n")
        features foreach {
          case (k, v) =>
            val padding = " " * (50 - featureName(k).size)
            output.write(f"${featureName(k)}$padding$v%.2f\n")
        }
        output.write("\n")
    }
    output.close
  }
}

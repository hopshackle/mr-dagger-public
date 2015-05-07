package coref.ml

import scala.util.Random
import scala.collection.Map

/**
 * Created by narad on 8/8/14.
 */
case class Prediction[T](label2score: Map[T, Double], entropy: Double = 0.0) {


  def randomMaxLabel(random: Random) = {
    assert(maxLabels.nonEmpty)
    val r = random.nextInt(maxLabels.size)
    maxLabels(r)
  }

  lazy val maxScore: Double = label2score.maxBy(_._2)._2

  lazy val maxLabels: Seq[T] = label2score.toSeq.filter(_._2 == maxScore).map(_._1)

  lazy val maxLabel = maxLabels.head

  override def toString = {
    "Prediction:\n" +
      label2score.keys.map { k =>
        "  %s: %f".format(k, label2score(k))
      }
  }
}
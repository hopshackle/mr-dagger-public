package dagger.util

/**
 * Created by narad on 4/2/15.
 */
case class EvalContainer(correct: Int = 0, predicted: Int = 0, gold: Int = 0) {

  def f1: Double = {
    if (correct == 0) return 0.0
    val mprec = correct * 1.0 / predicted
    val mrec = correct * 1.0 / gold
    if (mprec + mrec == 0) {
      0.0
    }
    else {
      val mf1 = 2 * (mprec * mrec) / (mprec + mrec)
      mf1
    }
  }

  def +(other: EvalContainer): EvalContainer = {
    EvalContainer(correct + other.correct, predicted + other.predicted, gold + other.gold)
  }
}
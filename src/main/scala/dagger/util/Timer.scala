package coref.util

/**
 * Created with IntelliJ IDEA.
 * User: narad
 * Date: 5/26/14
 * Time: 12:09 PM
 */

class Timer {
  private var startTime = 0.0
  private var stopTime = 0.0

  def elapsedTime() = {
    if (stopTime > 0.0) {
      stopTime - startTime
    }
    else {
      System.currentTimeMillis() - startTime
    }
  }

  def start() = startTime = System.currentTimeMillis()

  def stop() = stopTime = System.currentTimeMillis()

  override def toString = toString(divisor = 1.0)

  def toString(divisor: Double): String = {
    if (divisor == 0) return "-NA-"
    val etime = elapsedTime() / (divisor * 1000)
    if (etime > 60)
      "%1.1fm".format(etime/60)
    else
      "%1.4fs".format(etime)
  }

}

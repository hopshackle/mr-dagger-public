package dagger.util

/**
 * Created by narad on 4/30/15.
 */
object StringBuildSpeedTest extends App {

  test("FORMAT", 1000000)
  test("APPEND", 1000000)

  def test(method: String, rounds: Int) = {
    val start = System.currentTimeMillis()
    val index = new HashIndex(n = 3)
    for (i <- 1 to rounds) {
      if (method == "APPEND") {
//        index.index(str)
        val str = "test-string-" + "cat"
      }
      else {
        val str = "test-string-%s".format("cat")
 //       index.index(str)
      }
    }
    val end = System.currentTimeMillis()
    println("Test of method " + method + " concluded in " + (end - start))
  }

}

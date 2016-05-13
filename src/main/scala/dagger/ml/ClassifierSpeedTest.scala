package dagger.ml

class ClassifierSpeedTest[A](classifier: MultiClassClassifier[A], instances: Seq[Instance[A]]) {

  def runTest: Long = {
    val start = System.currentTimeMillis
    instances foreach classifier.predict
    val end = System.currentTimeMillis
    end - start
  }

}
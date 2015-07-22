package dagger.core

import dagger.ml.AROWOptions

/**
 * Created by narad on 10/18/14.
 */
class DAGGEROptions(args: Array[String]) extends AROWOptions(args) {

  lazy val ACTIVE_LEARNING = getBoolean("--active.learning", default = false)

  lazy val ACTIVE_THRESHOLD = getDouble("--active.threshold", 0.9)

  lazy val APPROXIMATE_LOSS = getBoolean("--approx.loss", false)

  lazy val CACHING = getBoolean("--cache", false)

  // Verifies that the first round DAGGER predictions reproduce the data
  // using the data structure [D]'s equality method.
  lazy val CHECK_ORACLE = getBoolean("--check.oracle", default = false)

  lazy val CLASSIFIER = getString("--classifier", "AROW") // {AROW, PASSIVE_AGGRESSIVE, PERCEPTRON}

  lazy val DAGGER_ITERATIONS = getInt("--dagger.iterations", 10)

  var DAGGER_OUTPUT_PATH = getString("--dagger.output.path", "./")

  lazy val DAGGER_PRINT_INTERVAL = getInt("--dagger.print.interval", 100)

  lazy val DAGGER_SERIALIZE_FILE = getString("--dagger.serialize.file", "serialization.txt")

  lazy val DEV_FILE = getString("--dev.file")

  lazy val DISCARD_OLD_INSTANCES = getBoolean("--discard.old.instances", default = false)

  lazy val EXPERT_APPROXIMATION = getBoolean("--expert.approx", false)

  lazy val MODEL_FILE = getString("--model.file", "model")

  lazy val NUM_CORES = getInt("--num.cores", 1)

  lazy val NUM_SAMPLES = getInt("--samples", 3)

  lazy val POLICY_DECAY = getDouble("--policy.decay", 0.2)

  lazy val SERIALIZE = getBoolean("--serialize", default = false)
  
  lazy val DEBUG = getBoolean("--debug", default = false)
  
  lazy val UNROLL_EXPERT_FOR_LOSS = getBoolean("--unrollExpert", default = false)
  
  lazy val ORACLE_LOSS = getBoolean("--oracleLoss", default = false)

  override def toString = {
    "DAGGER Options:\n" +
      "  DAGGER Iterations: %d\n".format(DAGGER_ITERATIONS) +
      "  DAGGER Dev file: %s\n".format(DEV_FILE) +
      "  DAGGER Classifier: %s\n".format(CLASSIFIER) +
      "  Discard instances each iteration: %s\n".format(DISCARD_OLD_INSTANCES) +
      "  Use expert approximation? %s\n".format(EXPERT_APPROXIMATION) +
      "  Approximate loss directly? %s\n".format(APPROXIMATE_LOSS) +
      "  Number of samples: %d\n".format(NUM_SAMPLES) +
      "  Policy decay rate: %f\n".format(POLICY_DECAY) +
      "  Number of cores: %d\n".format(NUM_CORES)
      super.toString
  }

}

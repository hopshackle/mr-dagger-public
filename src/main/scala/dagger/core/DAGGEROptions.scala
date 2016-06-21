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

  lazy val DAGGER_ITERATIONS = getInt("--dagger.iterations", 10)

  var DAGGER_OUTPUT_PATH = getString("--dagger.output.path", "./")

  lazy val DAGGER_PRINT_INTERVAL = getInt("--dagger.print.interval", 1)

  lazy val DAGGER_SERIALIZE_FILE = getString("--dagger.serialize.file", "serialization.txt")

  lazy val DEV_FILE = getString("--dev.file")

  lazy val DISCARD_OLD_INSTANCES = getBoolean("--discard.old.instances", default = false)

  lazy val EXPERT_APPROXIMATION = getBoolean("--expert.approx", false)

  lazy val MODEL_FILE = getString("--model.file", "model")

  lazy val NUM_CORES = getInt("--num.cores", 1)

  lazy val NUM_SAMPLES = getInt("--samples", 1)

  lazy val POLICY_DECAY = getDouble("--policy.decay", 0.2)

  lazy val SERIALIZE = getBoolean("--serialize", default = false)

  lazy val DEBUG = getBoolean("--debug", default = false)
  
  lazy val DETAIL_DEBUG = getBoolean("--detail", default = false)

  lazy val UNROLL_EXPERT_FOR_LOSS = getBoolean("--unrollExpert", default = false)

  var ORACLE_LOSS = getBoolean("--oracleLoss", default = false)

  lazy val ALGORITHM = getString("--algorithm", default = "Dagger")

  lazy val INITIAL_EXPERT_PROB = getDouble("--initialExpertProb", default = 1.0)
  
  lazy val PLOT_LOSS_PER_ITERATION = getBoolean("--plotLoss", default = false)

  var MAX_ACTIONS = getInt("--maxActions", default = 300)
  
  val ACTIONS_PER_SIZE = getInt("--actionsPerSize", default = 5)
  
  val WRITE_NAIVE_VAL_SCORES = getBoolean("--smatchScores", false)
  
  lazy val REDUCED_ACTION_SPACE = getBoolean("--reducedActions", false)
  
  lazy val ROLLOUT_THRESHOLD = getDouble("--threshold", 0.0)
  
  lazy val ROLLOUT_LIMIT = getInt("--rolloutLimit", 100)
  
  var USE_EXPERT_ON_ROLLOUT_AFTER = getInt("--expertAfter", 300)
  
  lazy val EXPERT_HORIZON_INCREMENT = getInt("--expertHorizonInc", 0)
  
  lazy val PREVIOUS_ITERATIONS_TO_USE = getInt("--previousTrainingIter", 100)
  
  lazy val BINARY_LOSS = getBoolean("--binaryLoss", false)
  
  lazy val COACHING_LAMBDA = getDouble("--coachingLambda", 0.0)
  
  lazy val RETRAIN_EACH_CLASSIFIER = getBoolean("--dropClassifier", false)
  
  lazy val EXPERT_HORIZON_ROLLOUT = getBoolean("--expertHorizon", false)

  lazy val MAX_TRAINING_SIZE = getInt("--maxTrainingSize", 100)
  
  lazy val MIN_TRAINING_SIZE = getInt("--minTrainingSize", 100)
  
  lazy val TRAINING_SIZE_INC = getInt("--trainingSizeInc", 10)
  
  lazy val LOG_TRAINING_STATS = getBoolean("--logTrainingStats", true)
  
  lazy val WRITE_FEATURE_INDEX = getBoolean("--writeFeatureIndex", false)
  
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

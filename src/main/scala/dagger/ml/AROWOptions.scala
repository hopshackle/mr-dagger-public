package dagger.ml

import dagger.util.ArgParser

/**
 * Created by narad on 10/16/14.
 */
class AROWOptions(args: Array[String]) extends ArgParser(args) {

  lazy val AROW_PRINT_INTERVAL = getInt("arow.print.interval", 100000)
  // A feature is considered rare, and may be removed if it occurs
  // less than this number of times in the training set.
  lazy val RARE_FEATURE_COUNT = getInt("--rare.feat.count", 0)

  lazy val RANDOM_SEED = getInt("--random.seed", 1)

  lazy val RATE = getDouble("--rate", 0.1)

  lazy val SHUFFLE = getBoolean("--shuffle", default = false)

  lazy val SMOOTHING = getDouble("--arow.smoothing", 1.0)

  var TRAIN_ITERATIONS = getInt("--arow.iterations", 50)

  lazy val TUNE_REGULARIZER = getBoolean("--tune.regularizer", default = false)

  lazy val AVERAGING = getBoolean("--average", default = false)

  lazy val VERBOSE = getBoolean("--verbose", default = false)

  override def toString = {
    "AROW OPTIONS:\n" +
    "  AROW min count for rare feats: %d\n".format(RARE_FEATURE_COUNT) +
    "  AROW train iterations: %d\n".format(TRAIN_ITERATIONS) +
    "  AROW tune regularizer: %s\n".format(TUNE_REGULARIZER) +
    "  AROW average params: %s\n".format(AVERAGING) +
    "  AROW shuffle data? %s\n".format(SHUFFLE) +
    "  Verbose? %s\n".format(VERBOSE)
  }
}


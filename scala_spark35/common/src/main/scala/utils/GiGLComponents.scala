package common.utils

object GiGLComponents extends Enumeration {
  val ConfigValidator, ConfigPopulator, DataPreprocessor, SubgraphSampler, SplitGenerator, Trainer,
    Inferencer, PostProcessor = Value
}

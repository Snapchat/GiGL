package splitgenerator.lib

object Types {
  object DatasetSplits extends Enumeration {
    type DatasetSplit = Value
    val TRAIN, TEST, VAL, OTHER = Value
  }

  object LinkUsageTypes extends Enumeration {
    type LinkUsageType = Value
    val MESSAGE, SUPERVISION, MESSAGE_AND_SUPERVISION = Value
  }

  /**
    * A link split is a pairing of
      - dataset split (train/val/test) and
      - edge usage (message/supervision)
      See http://web.stanford.edu/class/cs224w/slides/09-theory.pdf for details.
    *
    * @param datasetSplit
    * @param edgeUsage
    */
  case class LinkSplitType(
    datasetSplit: DatasetSplits.DatasetSplit,
    edgeUsage: LinkUsageTypes.LinkUsageType)

  case class SplitOutputPaths(
    trainPath: String,
    valPath: String,
    testPath: String) {
    def getPath(datasetSplit: DatasetSplits.DatasetSplit): String = {
      datasetSplit match {
        case DatasetSplits.TRAIN => trainPath
        case DatasetSplits.VAL   => valPath
        case DatasetSplits.TEST  => testPath
      }
    }
  }

  case class SplitSubsamplingRatio(
    trainSubsamplingRatio: Float,
    valSubsamplingRatio: Float,
    testSubsamplingRatio: Float) {
    def shouldSubsample: Boolean = {
      (trainSubsamplingRatio < 1.0f) || (valSubsamplingRatio < 1.0f) || (testSubsamplingRatio < 1.0f)
    }
  }
}

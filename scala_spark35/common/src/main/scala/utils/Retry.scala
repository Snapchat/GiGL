package common.utils

import scala.concurrent.duration.FiniteDuration

object Retry {

  /**
     * Usage:
     * val result = Retry.retry(
     *   retries=retries,
     *   initialDelay=FiniteDuration(1, "second"),
     *   backoffMultiplier=2,
     *   maxDelay=Some(FiniteDuration(4, "second"))
     *  ) {
     *   // code block to retry
     * }
     *
     * @param maxRetries the maximum number of retries
     * @param initialDelay the initial delay before the first retry
     * @param maxDelay the maximum delay between retries; default is None so the delay will keep increasing by the backoffMultiplier
     * @param backoffMultiplier the multiplier to increase the delay by after each retry; default is 1
     * @param retryableExceptions a list of exceptions that should be retried; by default, all exceptions are retried
     * @param block
     * @return
     */
  @annotation.tailrec
  def retry[T](
    retries: Int,
    initialDelay: FiniteDuration,
    maxDelay: Option[FiniteDuration] = None,
    backoffMultiplier: Int = 1,
    retryableExceptions: List[Class[_ <: Throwable]] = List(classOf[Exception]),
  )(
    fn: => T,
  ): T = {

    util.Try { fn } match { // Execute the function
      case util.Success(x) => x // Sucess -> return the result
      case util.Failure(e) => { // Failure -> retry if possible
        if (retries > 0 && retryableExceptions.exists(_.isInstance(e))) {
          val sleepTime = maxDelay match {
            case Some(max) => math.min(initialDelay.toMillis, max.toMillis)
            case None      => initialDelay.toMillis
          }
          println(s"Retry decorator will retry in ${sleepTime}ms; caught exception ${e}")
          Thread.sleep(sleepTime)
          retry(
            retries = retries - 1,
            initialDelay = initialDelay * backoffMultiplier,
            maxDelay = maxDelay,
            backoffMultiplier = backoffMultiplier,
            retryableExceptions = retryableExceptions,
          )(fn)
        } else {
          throw e
        }
      }
    }
  }

}

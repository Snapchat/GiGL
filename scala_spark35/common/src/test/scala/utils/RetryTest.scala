import common.utils.Retry
import org.scalatest.funsuite.AnyFunSuite

import scala.concurrent.duration.FiniteDuration

class RetryTest extends AnyFunSuite {

  test("retry should return result of the a successful block") {
    val result = Retry.retry(retries = 3, initialDelay = FiniteDuration(1, "second")) {
      1
    }
    assert(result == 1)
  }

  test("retry the specified number of times on failure") {
    var count = 0
    val result = Retry.retry(retries = 3, initialDelay = FiniteDuration(1, "second")) {
      count += 1
      if (count < 3) {
        throw new Exception("test")
      }
      count
    }
    assert(result == 3)
  }

  test("do not retry if the exception is not in the retryable list") {
    var count = 0
    assertThrows[RuntimeException] {
      Retry.retry(
        retries = 3,
        initialDelay = FiniteDuration(1, "second"),
        retryableExceptions = List(classOf[IllegalArgumentException]),
      ) {
        count += 1
        if (count < 3) {
          throw new RuntimeException("test")
        }
        count
      }
    }
    assert(count == 1)
  }

  test("respect the maximum delay") {
    var count             = 0
    val retries           = 5
    val initialDelay      = FiniteDuration(1, "second")
    val backoffMultiplier = 2
    val maxDelay          = Some(FiniteDuration(4, "second"))
    val oneSecondInMillis = 1000

    val startTimeMillis = System.currentTimeMillis()
    // With backoffMultiplier=2, the delays will be 1s, 2s, 4s, 4s
    val minElapsedTimeMillis = (1 + 2 + 4 + 4) * oneSecondInMillis
    val maxElapsedTimeMillis = minElapsedTimeMillis + oneSecondInMillis // One second buffer

    val result = Retry.retry(
      retries = retries,
      initialDelay = FiniteDuration(1, "second"),
      backoffMultiplier = 2,
      maxDelay = Some(FiniteDuration(4, "second")),
    ) {
      count += 1
      if (count <= 4) {
        throw new Exception("test")
      }
      println("count: " + count)
      count
    }
    val elapsedTimeMillis = System.currentTimeMillis() - startTimeMillis
    assert(result == 5)
    assert(elapsedTimeMillis <= maxElapsedTimeMillis)
    assert(elapsedTimeMillis >= minElapsedTimeMillis)
  }

}

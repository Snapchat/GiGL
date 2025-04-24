package common.utils

import java.net.HttpURLConnection
import java.net.URL
import scala.io.Source
import scala.util.Failure
import scala.util.Success
import scala.util.Try

object Env {
  private def envHasRelevantGCPEnvVars: Boolean = {
    val gcpEnvVars = Seq("GOOGLE_VM_CONFIG_LOCK_FILE")
    gcpEnvVars.exists(envVar => sys.env.contains(envVar))
  }

  private def envCanAccessGCPMetadataServer: Boolean = {
    val url = new URL("http://metadata.google.internal/computeMetadata/v1/project/project-id")
    val connection = url.openConnection().asInstanceOf[HttpURLConnection]
    connection.setRequestMethod("GET")
    connection.setRequestProperty("Metadata-Flavor", "Google")

    Try(Source.fromInputStream(connection.getInputStream).mkString) match {
      case Success(_) =>
        true
      case Failure(_) =>
        false
    }
  }

  def isRunningOnGCP: Boolean = {
    envHasRelevantGCPEnvVars || envCanAccessGCPMetadataServer
  }
}

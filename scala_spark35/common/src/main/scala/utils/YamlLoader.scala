package common.utils

import com.google.cloud.storage.BlobId
import com.google.cloud.storage.Storage
import com.google.cloud.storage.StorageOptions

object YamlLoader {
  def readGcsFileAsString(
    bucketName: String,
    objectName: String,
  ): String = {
    val storage: Storage = StorageOptions.getDefaultInstance.getService
    val blobId: BlobId   = BlobId.of(bucketName, objectName)
    val blob             = storage.get(blobId)
    val content          = new String(blob.getContent())
    content
  }

  def readYamlAsString(
    uri: String,
  ): String = {
    val dataYamlString = if (uri.startsWith("gs://")) {
      val uriParts       = uri.split("/")
      val bucketName     = uriParts(2)
      val objectName     = uriParts.slice(3, uriParts.length).mkString("/")
      val dataYamlString = readGcsFileAsString(bucketName, objectName)
      dataYamlString
    } else {
      val dataYamlString = scala.io.Source.fromFile(uri).getLines.mkString("\n")
      dataYamlString
    }
    dataYamlString
  }
}

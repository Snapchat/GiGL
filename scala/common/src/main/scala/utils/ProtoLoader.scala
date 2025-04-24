package common.utils

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.yaml.snakeyaml.Yaml
import scalapb.GeneratedMessage
import scalapb.GeneratedMessageCompanion
import scalapb.json4s.JsonFormat

import java.io.File
import java.io.FileInputStream

object ProtoLoader {
  def loadYamlStrToProto[A <: GeneratedMessage: GeneratedMessageCompanion](yaml: String): A = {
    // Parsing the YAML file with SnakeYAML - since Jackson Parser does not have Anchors and reference support
    val yml      = new Yaml()
    val mapper   = new ObjectMapper().registerModules(DefaultScalaModule)
    val yamlFile = new File(yaml)
    val yamlObj: Any = if (yamlFile.exists()) {
      val yamlFileStream = new FileInputStream(yamlFile)
      yml.loadAs(yamlFileStream, classOf[Any])
    } else {
      yml.loadAs(yaml, classOf[Any])
    }
    // Converting the YAML to Jackson YAML - since it has more flexibility
    val jsonString = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(yamlObj)
    val proto: A   = JsonFormat.fromJsonString[A](jsonString)
    proto
  }

  def populateProtoFromYaml[T <: GeneratedMessage: GeneratedMessageCompanion](
    uri: String,
    printYamlContents: Boolean = false,
  ): T = {
    val dataYamlString = YamlLoader.readYamlAsString(uri = uri)
    if (printYamlContents == true) { println(dataYamlString) }
    val proto: T = ProtoLoader.loadYamlStrToProto[T](dataYamlString)
    proto
  }
}

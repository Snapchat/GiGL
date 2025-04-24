import scala.io.Source

ThisBuild / scalaVersion := "2.12.15"
ThisBuild / organization := "snapchat.research.gbml"
ThisBuild / version      := "1.0"
// Enable semantic db so that scalafix can run the "OrganizeImports" rule.
ThisBuild / semanticdbEnabled := true
ThisBuild / semanticdbVersion := scalafixSemanticdb.revision

lazy val SPARK_31_TFRECORD_JAR_GCS_PATH: String = {
  val filePath = "../dep_vars.env"
  val source = Source.fromFile(filePath)
  try {
    source.getLines().find(_.startsWith("SPARK_31_TFRECORD_JAR_GCS_PATH=")) match {
      case Some(line) => line.split("=")(1).trim
      case None => throw new RuntimeException(s"SPARK_31_TFRECORD_JAR_GCS_PATH not found in $filePath")
    }
  } finally {
    source.close()
  }
}

// Dependencies ========================================
lazy val dependencies =
  new {
    val scalaParserCombinators = "org.scala-lang.modules" %% "scala-parser-combinators" % "1.1.2"

    // Needed for being able to compile and use protos from tools/scalapbc
    val scalapbRuntime = "com.thesamet.scalapb" %% "scalapb-runtime" % "0.11.12"

    // JSON + YAML parsing for ScalaPB
    val jackson     = "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.14.1"
    val scalapbJson = "com.thesamet.scalapb"         %% "scalapb-json4s"       % "0.11.1"
    val snakeyaml   = "org.yaml"                      % "snakeyaml"            % "1.16"

    // Needed for spark applications
    val sparkCore    = "org.apache.spark"     %% "spark-core"             % "3.1.3" % "provided"
    val sparkSql     = "org.apache.spark"     %% "spark-sql"              % "3.1.3" % "provided"
    val scalapbSpark = "com.thesamet.scalapb" %% "sparksql31-scalapb0_11" % "1.0.0"

    // Working with google cloud storage
    val gcs = "com.google.cloud" % "google-cloud-storage" % "2.16.0"

    // http request
    val http = "org.scalaj" %% "scalaj-http" % "2.4.2"

    // Test Dependencies
    val scalatest = "org.scalatest" %% "scalatest" % "3.2.11" % Test
    // Not included in fat jar during compile time due to dependency issues; injected through spark-submit at runtime
    // The jar file is built using Snap's fork of the Linkedin TfRecord Spark Connector.
    val tfRecordConnector =
      "com.linkedin.sparktfrecord" % "spark-tfrecord_2.12" % "0.5.0" % Test from SPARK_31_TFRECORD_JAR_GCS_PATH

    // nebula client
    val nebulaClient = "com.vesoft" % "client" % "3.6.1"

    val scalaLogging = "com.typesafe.scala-logging" %% "scala-logging" % "3.9.4"
  }

lazy val commonDependencies = Seq(
  dependencies.scalaParserCombinators,
  dependencies.scalapbRuntime,
  dependencies.jackson,
  dependencies.scalapbJson,
  dependencies.snakeyaml,
  dependencies.sparkCore,
  dependencies.sparkSql,
  dependencies.scalapbSpark,
  dependencies.gcs,
  dependencies.http,
  dependencies.scalatest,
  dependencies.scalaLogging,
  dependencies.nebulaClient,
)

// ======================================== Dependencies

// Settings ========================================
lazy val assemblySettings = Seq(
  // META-INF files are deprecated but when compileine a fat jar get marked as duplicates
  // We can safely discard the duplicates.
  assembly / assemblyMergeStrategy := {
    case PathList("META-INF", xs @ _*) => MergeStrategy.discard
    case x => {
      val oldStrategy = (assembly / assemblyMergeStrategy).value
      oldStrategy(x)
    }
  },
  // Spark ships with an old version of Google's Protocol Buffers runtime that is not compatible with the current version. In addition, it comes with incompatible versions of scala-collection-compat and shapeless. Therefore, we need to shade these libraries. Add the following to your build.sbt:
  assembly / assemblyShadeRules := Seq(
    ShadeRule.rename("com.google.protobuf.**" -> "shadeproto.@1").inAll,
    ShadeRule.rename("scala.collection.compat.**" -> "scalacompat.@1").inAll,
    ShadeRule.rename("shapeless.**" -> "shadeshapeless.@1").inAll,
  ),
  assembly / test := {}, // Disable tests when run sbt assembly
)

lazy val testSettings = Seq(
  Test / parallelExecution := false, // If true all test classes will run in parallel, and afterAll() method runs after each test class. [we don't want this in case of SparkSession, since a test that ends earliest stops Sparksession and other tests will error out due to missing SparkSession]
  Test / unmanagedSourceDirectories += baseDirectory.value / ".." / "common" / "src" / "test" / "scala" / "utils", // This adds the test utils directory to the test of the projects as well so that all tests can share the utils like SharedSparkSession.
)

lazy val scalacOpts = Seq(
  // Enable unused imports warnings so scalafix OrganizeImports can remove
  // unused imports.
  scalacOptions ++= Seq(
    // scalafix only supports scala 2.12 and up, it's not that previous scala versions
    // *also* use -WUnused:imports
    if (scalaVersion.value.startsWith("2.12")) {
      "-Ywarn-unused-import"
    } else {
      "-Wunused:imports"
    },
    // Skip formatting generated proto files.
    // We use dots instead of / here since forward slash may not work on Windows.
    "-P:semanticdb:exclude:snapchat.research.gbml.*scala",
  )
)

// ======================================== Settings

// Projects ========================================
lazy val common = (project in file("common"))
  .settings(
    name := "common",
    libraryDependencies ++= commonDependencies,
    scalacOpts,
  )
  .disablePlugins(AssemblyPlugin) // Disable assembly to jar

lazy val subgraph_sampler = (project in file("subgraph_sampler"))
  .settings(
    name := "subgraph_sampler",
    libraryDependencies ++= commonDependencies ++ Seq(dependencies.tfRecordConnector),
    assemblySettings,
    testSettings,
    scalacOpts,
  )
  .dependsOn(common)

lazy val split_generator = (project in file("split_generator"))
  .settings(
    name := "split_generator",
    libraryDependencies ++= commonDependencies ++ Seq(dependencies.tfRecordConnector),
    assemblySettings,
    testSettings,
    scalacOpts,
  )
  .dependsOn(common)

// ======================================== Projects

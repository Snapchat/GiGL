import scala.io.Source

ThisBuild / scalaVersion := "2.12.18"
ThisBuild / organization := "snapchat.research.gbml"
ThisBuild / version      := "1.0"
// Enable semantic db so that scalafix can run the "OrganizeImports" rule.
ThisBuild / semanticdbEnabled := true
ThisBuild / semanticdbVersion := scalafixSemanticdb.revision

lazy val SPARK_35_TFRECORD_JAR_GCS_PATH: String = {
  val filePath = "../dep_vars.env"
  val source = Source.fromFile(filePath)
  try {
    source.getLines().find(_.startsWith("SPARK_35_TFRECORD_JAR_GCS_PATH=")) match {
      case Some(line) => line.split("=")(1).trim
      case None => throw new RuntimeException(s"SPARK_35_TFRECORD_JAR_GCS_PATH not found in $filePath")
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
    val scalapbRuntime = "com.thesamet.scalapb" %% "scalapb-runtime" % "0.11.14"

    // JSON + YAML parsing for ScalaPB
    val jackson     = "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.14.1"
    val scalapbJson = "com.thesamet.scalapb"         %% "scalapb-json4s"       % "0.11.1"
    val snakeyaml   = "org.yaml"                      % "snakeyaml"            % "1.16"
    // Serialize/Deserialize JSON strings
    val upickle = "com.lihaoyi" %% "upickle" % "3.1.0"

    // Needed for spark applications
    val sparkCore    = "org.apache.spark"     %% "spark-core"             % "3.5.0" % "provided"
    val sparkSql     = "org.apache.spark"     %% "spark-sql"              % "3.5.0" % "provided"
    val scalapbSpark = "com.thesamet.scalapb" %% "sparksql35-scalapb0_11" % "1.0.4"

    // Working with google cloud storage
    val gcs = "com.google.cloud" % "google-cloud-storage" % "2.16.0"

    // http request
    val http = "org.scalaj" %% "scalaj-http" % "2.4.2"

    // GRPC dependencies
    val grpcNetty    = "io.grpc" % "grpc-netty"    % "1.51.0"
    val grpcProtobuf = "io.grpc" % "grpc-protobuf" % "1.51.0"
    val grpcStub     = "io.grpc" % "grpc-stub"     % "1.51.0"

    // Testing
    val scalatest = "org.scalatest" %% "scalatest" % "3.2.11" % Test
    // Not included in fat jar during compile time due to dependency issues; injected through spark-submit at runtime
    // TODO: (svij-sc) Find a common place to pull this jar uri from
    // The jar file is built using Snap's fork of the Linkedin TfRecord Spark Connector.
    val tfRecordConnector =
      "com.linkedin.sparktfrecord" % "spark-tfrecord_2.12" % "0.6.1" % Test from SPARK_35_TFRECORD_JAR_GCS_PATH

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
  dependencies.upickle,
  dependencies.sparkCore,
  dependencies.sparkSql,
  dependencies.scalapbSpark,
  dependencies.gcs,
  dependencies.http,
  dependencies.scalatest,
  dependencies.scalaLogging,
  dependencies.nebulaClient,
  dependencies.grpcNetty,
  dependencies.grpcProtobuf,
  dependencies.grpcStub,
)

lazy val commonDependencyOverrides = Seq(
  // custom overrides here
) ++ GCPDepsBOM.v26_2_0_deps

// ======================================== Dependencies

// Settings ========================================
lazy val assemblySettings = Seq(
  assembly / assemblyMergeStrategy := {
    // Correctly handle the META-INF/services/* files, which causes an issue when creating fat jars.
    // A service provider is identified by placing a provider-configuration file in the resource directory META-INF/services.
    // The file's name is the fully-qualified binary name of the service's type. The file contains a list of fully-qualified
    // binary names of concrete provider classes, one per line... For more info see:
    // https://docs.oracle.com/javase/7/docs/api/java/util/ServiceLoader.html#:~:text=A%20service%20provider%20is%20identified,provider%20classes%2C%20one%20per%20line.
    // This file is used by Java's ServiceLoader mechanism and needs to be properly merged to
    // include all available NameResolverProvider implementations. This feature is used as transitive dependencies
    // from the ones we import.
    case PathList("META-INF", "services", xs @ _*) => MergeStrategy.concat
    // Similar to above, this can cause some conflicts when creating fat jars when using GRPC.
    // Most jars for io.netty deps (many transative deps) contain a file called io.netty.versions.properties
    // This file may not be located inside `META-INF`; and thus may cause issues if multiple are included.
    // We discard duplicates.
    case x if x.contains("io.netty.versions.properties") => MergeStrategy.discard
    // Any other META-INF files can be safely discarded as they are not needed in the fat jar.
    case PathList("META-INF", xs @ _*) => MergeStrategy.discard
    case x => {
      val oldStrategy = (assembly / assemblyMergeStrategy).value
      oldStrategy(x)
    }
  },

  // Spark ships with an old version of Google's Protocol Buffers runtime that is not compatible with the current version.
  // In addition, it comes with incompatible versions of scala-collection-compat and shapeless.
  // Therefore, we need to shade these libraries.
  assembly / assemblyShadeRules := Seq(
    ShadeRule.rename("com.google.protobuf.**" -> "shadeproto.@1").inAll,
    ShadeRule.rename("scala.collection.compat.**" -> "scalacompat.@1").inAll,
    ShadeRule.rename("shapeless.**" -> "shadeshapeless.@1").inAll,
  ),
  assembly / test := {}, // Disable tests when run sbt assembly
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


lazy val testParallelExecutionSetting = Seq(
  Test / parallelExecution := false,
)

lazy val testUnmanagedSourceDirectoriesSetting = Seq(
  Test / unmanagedSourceDirectories += baseDirectory.value / ".." / "common" / "src" / "test" / "scala" / "testLibs",
)

lazy val testSettings = testParallelExecutionSetting ++ testUnmanagedSourceDirectoriesSetting

// ======================================== Settings

// Projects ========================================
lazy val common = (project in file("common"))
  .settings(
    name := "common",
    libraryDependencies ++= commonDependencies ++ Seq(dependencies.tfRecordConnector),
    dependencyOverrides ++= commonDependencyOverrides,
    testParallelExecutionSetting,
    scalacOpts,
    assemblySettings,
  )

lazy val subgraph_sampler = (project in file("subgraph_sampler"))
  .settings(
    name := "subgraph_sampler",
    libraryDependencies ++= commonDependencies ++ Seq(dependencies.tfRecordConnector),
    dependencyOverrides ++= commonDependencyOverrides,
    assemblySettings,
    testSettings,
    scalacOpts,
  )
  .dependsOn(common)

lazy val split_generator = (project in file("split_generator"))
  .settings(
    name := "split_generator",
    libraryDependencies ++= commonDependencies ++ Seq(dependencies.tfRecordConnector),
    dependencyOverrides ++= commonDependencyOverrides,
    assemblySettings,
    testSettings,
    scalacOpts,
  )
  .dependsOn(common)

// ======================================== Projects

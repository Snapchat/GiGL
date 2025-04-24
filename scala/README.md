## GIGL Spark/Scala Repo

### Building and Testing the project

Run `make install_deps` first if you haven't to install scala & spark. cd inside the `scala` directory. To compile all
projects in the repo and generate the jar files, run:

```
sbt assembly
```

To assemble a specific project:

```
sbt subgraph_sampler/assembly
```

To run all tests:

```
sbt test
```

Similarly to run a specific test suite:

```
sbt subgraph_sampler/test
```

To clean up all target files:

```
make clean_build_files
```

### Running spark jobs locally

Please check the Makefile for commands to run the spark jobs locally. The jobs makes use of the mocked assets from the
directory

```
common/src/test/assets
```

#### Set log level

To silence the worker logs

1. Create log4j.properties file from template, under `/scala` dir, do
   `cp ../tools/scala/spark-3.1.3-bin-hadoop3.2/conf/log4j.properties.template ../tools/scala/spark-3.1.3-bin-hadoop3.2/conf/log4j.properties`
1. Update the first line in `log4j.properties` to `log4j.rootCategory=WARN, console`

Note: Mocked assets are generated using the dataset asset mocking suite (in `python/gigl/src/mocking/`)

### How to build and deploy spark-tfrecord package used in the Spark Jobs

Note: remember to have local deps for developing installed by running `make install_deps`. See main README.md for more
details.

We make use of the Spark [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) Connector provided by the
Linkedin repo [linkedin/spark-tfrecord](https://github.com/linkedin/spark-tfrecord). We deploy and maintain our own
copies off the jar since not all sbt/scala vers are available on Maven Central, etc.

To build:

First clone the repo, then cd into directory.

Install maven if not already installed:

```
Linux:
sudo apt-get install maven

OSX:
brew install maven
```

Build with maven (specific scala and spark versions can be found in `build.sbt` file in our repo)

```
mvn -Pscala-2.12 clean install -Dspark.version=3.2.0
```

Copy to GCS and deploy:

```
gsutil cp target/spark-tfrecord_2.12-0.5.0.jar gs://$YOUR_BUCKET/your/path/to/snap-spark-custom-tfrecord_2.12-{version_number}.jar
```

Note: Snap currently hosts these in the `public-gigl` GCS bucket.

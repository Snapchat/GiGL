include ../dep_vars.env

# Ensure that path can be resolved globally to the jar
# In this case, MAKEFILE_DIRS_PARENT_PATH resolves to the root of the repo
MAKEFILE_DIRS_PARENT_PATH := $(realpath $(dir $(realpath $(firstword $(MAKEFILE_LIST))))/..)
ROOTED_SPARK_35_TFRECORD_JAR_LOCAL_PATH:=$(MAKEFILE_DIRS_PARENT_PATH)/$(SPARK_35_TFRECORD_JAR_LOCAL_PATH)

DATAPROC_CLUSTER_NAME:=YOUR_CLUSTER_NAME
TMP_GIGL_DEV_GCS_BUCKET_NAME:=$(shell echo $$TMP_GIGL_DEV_GCS_BUCKET_NAME)

CURRENT_USER := $(shell whoami)
CURRENT_DATETIME := $(shell date "+%Y%m%d_%H%M%S")
TMP_GIGL_DEV_GCS_PATH_PREFIX:=gs://$(TMP_GIGL_DEV_GCS_BUCKET_NAME)/$(CURRENT_USER)/$(CURRENT_DATETIME)


# Below is the location sbt assemply will generate the jar to
LOCAL_SGS_ASSEMBLY_JAR_PATH:=subgraph_sampler/target/scala-2.12/subgraph_sampler-assembly-1.0.jar
# Below is the GCS location we will copy the local assembled jar to
GCS_SGS_ASSEMBLY_JAR_PATH:=$(TMP_GIGL_DEV_GCS_PATH_PREFIX)/jars/subgraph_sampler-assembly-1.0.jar

# Below is the location sbt assemply will generate the jar to
LOCAL_SPLITGEN_ASSEMBLY_JAR_PATH:=split_generator/target/scala-2.12/split_generator-assembly-1.0.jar
# Below is the GCS location we will copy the local assembled jar to
GCS_SPLITGEN_ASSEMBLY_JAR_PATH:=$(TMP_GIGL_DEV_GCS_PATH_PREFIX)/jars/split-generator-assembly-1.0.jar

LOCAL_TEST_RESOURCE_CONFIG_PATH:=common/src/test/assets/resource_config.yaml
# Below is the GCS location we will copy the local resource config to
GCS_TEST_RESOURCE_CONFIG_PATH:=$(TMP_GIGL_DEV_GCS_PATH_PREFIX)/resource_configs/resource_config.yaml

# Replace below with your own frozen task config path
LOCAL_TEST_FROZEN_TASK_CONFIG_PATH:=common/src/test/assets/subgraph_sampler/node_anchor_based_link_prediction/frozen_gbml_config.yaml
# GraphDB Version # LOCAL_TEST_FROZEN_TASK_CONFIG_PATH:=common/src/test/assets/subgraph_sampler/heterogeneous/node_anchor_based_link_prediction/frozen_gbml_config_graphdb_dblp_local.yaml
# Below is the GCS location we will copy the local frozen task config to
GCS_TEST_FROZEN_TASK_CONFIG_PATH:=$(TMP_GIGL_DEV_GCS_PATH_PREFIX)/frozen_task_configs/frozen_gbml_config.yaml


_assert_tmp_gigl_dev_gcs_bucket_name_set:
ifndef TMP_GIGL_DEV_GCS_BUCKET_NAME
	$(error Please set env var TMP_GIGL_DEV_GCS_BUCKET_NAME to your GCS bucket name)
endif

unittest:
	sbt "test:testOnly *$(test-name)"

# TODO: (svij) Provide instructions on how a debug cluster can be created
spark_run_sgs_on_current_machine:
	sbt subgraph_sampler/assembly
	../tools/scala/spark-3.5.0-bin-hadoop3/bin/spark-submit \
		--class Main \
		--master local \
		--jars $(ROOTED_SPARK_35_TFRECORD_JAR_LOCAL_PATH) \
		$(LOCAL_SGS_ASSEMBLY_JAR_PATH) \
		$(LOCAL_TEST_FROZEN_TASK_CONFIG_PATH) local_sgs_job $(LOCAL_TEST_RESOURCE_CONFIG_PATH)

spark_run_splitgen_on_current_machine:
	sbt split_generator/assembly
	../tools/scala/spark-3.5.0-bin-hadoop3/bin/spark-submit \
		--class Main \
		--master local \
		--jars $(ROOTED_SPARK_35_TFRECORD_JAR_LOCAL_PATH) \
		$(LOCAL_SPLITGEN_ASSEMBLY_JAR_PATH) \
		local_splitgen_job $(LOCAL_TEST_FROZEN_TASK_CONFIG_PATH) $(LOCAL_TEST_RESOURCE_CONFIG_PATH)

spark_run_sgs_on_cluster: _assert_tmp_gigl_dev_gcs_bucket_name_set
	sbt subgraph_sampler/assembly
	gsutil cp $(LOCAL_SGS_ASSEMBLY_JAR_PATH) $(GCS_SGS_ASSEMBLY_JAR_PATH)
	gsutil cp $(LOCAL_TEST_RESOURCE_CONFIG_PATH) $(GCS_TEST_RESOURCE_CONFIG_PATH)
	gsutil cp $(LOCAL_TEST_FROZEN_TASK_CONFIG_PATH) $(GCS_TEST_FROZEN_TASK_CONFIG_PATH)
	gcloud dataproc jobs submit spark \
	--cluster $(DATAPROC_CLUSTER_NAME) \
	--region us-central1 \
	--jar $(GCS_SGS_ASSEMBLY_JAR_PATH) \
	--jars $(SPARK_35_TFRECORD_JAR_GCS_PATH) \
	-- $(GCS_TEST_FROZEN_TASK_CONFIG_PATH) cluster_sgs_job $(GCS_TEST_RESOURCE_CONFIG_PATH)


spark_run_splitgen_on_cluster: _assert_tmp_gigl_dev_gcs_bucket_name_set
	sbt split_generator/assembly
	gsutil cp $(LOCAL_SPLITGEN_ASSEMBLY_JAR_PATH) $(GCS_SPLITGEN_ASSEMBLY_JAR_PATH)
	gsutil cp $(LOCAL_TEST_RESOURCE_CONFIG_PATH) $(GCS_TEST_RESOURCE_CONFIG_PATH)
	gsutil cp $(LOCAL_TEST_FROZEN_TASK_CONFIG_PATH) $(GCS_TEST_FROZEN_TASK_CONFIG_PATH)
	gcloud dataproc jobs submit spark \
	--cluster $(DATAPROC_CLUSTER_NAME) \
	--region us-central1 \
	--jar $(GCS_SPLITGEN_ASSEMBLY_JAR_PATH) \
	--jars $(SPARK_35_TFRECORD_JAR_GCS_PATH) \
	-- cluster_splitgen_job $(GCS_TEST_FROZEN_TASK_CONFIG_PATH) $(GCS_TEST_RESOURCE_CONFIG_PATH)

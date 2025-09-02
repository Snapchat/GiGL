include dep_vars.env

SHELL := /bin/bash
CONDA_ENV_NAME=gnn
PYTHON_VERSION=3.9
PIP_VERSION=25.0.1
DATE:=$(shell /bin/date "+%Y%m%d_%H%M")

# GIT HASH, or empty string if not in a git repo.
GIT_HASH?=$(shell git rev-parse HEAD 2>/dev/null || "")
PWD=$(shell pwd)


# You can override GIGL_PROJECT by setting it in your environment i.e.
# adding `export GIGL_PROJECT=your_project` to your shell config (~/.bashrc, ~/.zshrc, etc.)
GIGL_PROJECT?=external-snap-ci-github-gigl
GIGL_DOCKER_ARTIFACT_REGISTRY?=us-central1-docker.pkg.dev/${GIGL_PROJECT}/gigl-base-images
DOCKER_IMAGE_DATAFLOW_RUNTIME_NAME:=${GIGL_DOCKER_ARTIFACT_REGISTRY}/src-cpu-dataflow
DOCKER_IMAGE_MAIN_CUDA_NAME:=${GIGL_DOCKER_ARTIFACT_REGISTRY}/src-cuda
DOCKER_IMAGE_MAIN_CPU_NAME:=${GIGL_DOCKER_ARTIFACT_REGISTRY}/src-cpu
DOCKER_IMAGE_DEV_WORKBENCH_NAME:=${GIGL_DOCKER_ARTIFACT_REGISTRY}/dev-workbench

DOCKER_IMAGE_DATAFLOW_RUNTIME_NAME_WITH_TAG?=${DOCKER_IMAGE_DATAFLOW_RUNTIME_NAME}:${DATE}
DOCKER_IMAGE_MAIN_CUDA_NAME_WITH_TAG?=${DOCKER_IMAGE_MAIN_CUDA_NAME}:${DATE}
DOCKER_IMAGE_MAIN_CPU_NAME_WITH_TAG?=${DOCKER_IMAGE_MAIN_CPU_NAME}:${DATE}
DOCKER_IMAGE_DEV_WORKBENCH_NAME_WITH_TAG?=${DOCKER_IMAGE_DEV_WORKBENCH_NAME}:${DATE}

PYTHON_DIRS:=.github/scripts examples python scripts testing
PY_TEST_FILES?="*_test.py"
# You can override GIGL_TEST_DEFAULT_RESOURCE_CONFIG by setting it in your environment i.e.
# adding `export GIGL_TEST_DEFAULT_RESOURCE_CONFIG=your_resource_config` to your shell config (~/.bashrc, ~/.zshrc, etc.)
GIGL_TEST_DEFAULT_RESOURCE_CONFIG?=${PWD}/deployment/configs/unittest_resource_config.yaml
# Default path for compiled KFP pipeline
GIGL_E2E_TEST_COMPILED_PIPELINE_PATH:=/tmp/gigl/pipeline_${DATE}_${GIT_HASH}.yaml

GIT_BRANCH:=$(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")

# If we're in a git repo, then find only the ".md" files in our repo to format, else we format everything ".".
# We do this because some of our dependencies (Spark) include md files,
# but since we don't push those dependenices (or their documentation) to git,
# then when we *check* the format of those files, we will fail.
# Thus, we only want to format the Markdown files that we explicitly include in our repo.
MD_FILES:=$(shell if [ ! ${GIT_BRANCH} ]; then echo "."; else git ls-tree --name-only -r ${GIT_BRANCH} . | grep ".md"; fi;)


get_ver_hash:
	# Fetches the git commit hash and stores it in `$GIT_COMMIT`
	git diff --quiet || { echo Branch is dirty, please commit changes and ensure branch is clean; exit 1; }
	$(eval GIT_COMMIT=$(shell git log -1 --pretty=format:"%H"))

initialize_environment:
	conda create -y --override-channels --channel conda-forge --name ${CONDA_ENV_NAME} python=${PYTHON_VERSION} pip=${PIP_VERSION} pip-tools
	@echo "If conda environment was successfully installed, ensure to activate it and run \`make install_dev_deps\` or \`make install_deps\` to complete setup"

clean_environment:
	if [ "${CONDA_DEFAULT_ENV}" == "${CONDA_ENV_NAME}" ]; then \
		pip uninstall -y -r <(pip freeze); \
	else \
		echo Change your local env to dev first.; \
	fi

reset_environment: generate_cpu_hashed_requirements clean_environment install_deps

rebuild_dev_environment:
	conda deactivate
	conda remove --name ${CONDA_ENV_NAME} --all -y
	make initialize_environment
	conda activate ${CONDA_ENV_NAME}
	make install_dev_deps

check_if_valid_env:
	#@command -v docker >/dev/null 2>&1 || { echo >&2 "docker is required but it's not installed.  Aborting."; exit 1; }
	@command -v gsutil >/dev/null 2>&1 || { echo >&2 "gsutil is required but it's not installed.  Aborting."; exit 1; }
	@python --version | grep -q "Python ${PYTHON_VERSION}" || (echo "Python version is not 3.9" && exit 1)


# if developing, you need to install dev deps instead
install_dev_deps: check_if_valid_env
	gcloud auth configure-docker us-central1-docker.pkg.dev
	bash ./requirements/install_py_deps.sh --dev
	bash ./requirements/install_scala_deps.sh
	pip install -e ./python/
	pre-commit install --hook-type pre-commit --hook-type pre-push


# Production environments, if you are developing use `make install_dev_deps` instead
install_deps:
	gcloud auth configure-docker us-central1-docker.pkg.dev
	bash ./requirements/install_py_deps.sh
	bash ./requirements/install_scala_deps.sh
	pip install -e ./python/

# Can only be run on an arm64 mac, otherwise generated hashed req file will be wrong
generate_mac_arm64_cpu_hashed_requirements:
	pip-compile -v --allow-unsafe --generate-hashes --no-emit-index-url --resolver=backtracking \
	--output-file=requirements/darwin_arm64_requirements_unified.txt \
	--extra torch25-cpu --extra transform --extra experimental \
	./python/pyproject.toml

# Can only be run on an arm64 mac, otherwise generated hashed req file will be wrong.
generate_dev_mac_arm64_cpu_hashed_requirements:
	pip-compile -v --allow-unsafe --generate-hashes --no-emit-index-url --resolver=backtracking \
	--output-file=requirements/dev_darwin_arm64_requirements_unified.txt \
	--extra torch25-cpu --extra transform --extra dev --extra experimental \
	./python/pyproject.toml

# Can only be run on linux, otherwise generated hashed req file will be wrong.
generate_linux_cpu_hashed_requirements:
	pip-compile -v --allow-unsafe --generate-hashes --no-emit-index-url --resolver=backtracking \
	--output-file=requirements/linux_cpu_requirements_unified.txt \
	--extra torch25-cpu --extra transform --extra experimental \
	./python/pyproject.toml

# Can only be run on linux, otherwise generated hashed req file will be wrong.
generate_dev_linux_cpu_hashed_requirements:
	pip-compile -v --allow-unsafe --generate-hashes --no-emit-index-url --resolver=backtracking \
	--output-file=requirements/dev_linux_cpu_requirements_unified.txt \
	--extra torch25-cpu --extra transform --extra dev --extra experimental \
	./python/pyproject.toml

# Can only be run on linux, otherwise generated hashed req file will be wrong.
generate_linux_cuda_hashed_requirements:
	pip-compile  -v --allow-unsafe --generate-hashes --no-emit-index-url --resolver=backtracking \
	--output-file=requirements/linux_cuda_requirements_unified.txt \
	--extra torch25-cuda-121 --extra transform --extra experimental \
	./python/pyproject.toml

# Can only be run on linux, otherwise generated hashed req file will be wrong.
generate_dev_linux_cuda_hashed_requirements:
	pip-compile -v --allow-unsafe --generate-hashes --no-emit-index-url --resolver=backtracking \
	--output-file=requirements/dev_linux_cuda_requirements_unified.txt \
	--extra torch25-cuda-121 --extra transform --extra dev --extra experimental \
	./python/pyproject.toml

# These are a collection of tests that are run before anything is installed using tools available on host.
# May include tests that check the sanity of the repo state i.e. ones that may even cause the failure of
# installation scripts
precondition_tests:
	python testing/dep_vars_check.py


run_api_test:
	cd testing/api_test && make run_api_test


assert_yaml_configs_parse:
	python testing/assert_yaml_configs_parse.py -d .

# Set PY_TEST_FILES=<TEST_FILE_NAME_GLOB> to test a specifc file.
# Ex. `make unit_test_py PY_TEST_FILES="eval_metrics_test.py"`
# By default, runs all tests under python/tests/unit.
# See the help text for "--test_file_pattern" in python/tests/test_args.py for more details.
unit_test_py: clean_build_files_py type_check
	( cd python ; \
	python -m tests.unit.main \
		--env=test \
		--resource_config_uri=${GIGL_TEST_DEFAULT_RESOURCE_CONFIG} \
		--test_file_pattern=$(PY_TEST_FILES) \
	)

unit_test_scala: clean_build_files_scala
	( cd scala; sbt test )
	( cd scala_spark35 ; sbt test )

# Runs unit tests for Python and Scala
# Asserts Python and Scala files are formatted correctly.
# Asserts YAML configs can be parsed.
# TODO(kmonte): We shouldn't be making assertions about format in unit_test, but we do so that
# we don't need to setup the dev environment twice in jenkins.
# Eventually, we should look into splitting these up.
# We run `make check_format` separately instead of as a dependent make rule so that it always runs after the actual testing.
# We don't want to fail the tests due to non-conformant formatting during development.
unit_test: precondition_tests unit_test_py unit_test_scala

check_format_py:
	autoflake --check --config python/pyproject.toml ${PYTHON_DIRS}
	isort --check-only --settings-path=python/pyproject.toml ${PYTHON_DIRS}
	black --check --config=python/pyproject.toml ${PYTHON_DIRS}

check_format_scala:
	( cd scala; sbt "scalafmtCheckAll; scalafixAll --check"; )
	( cd scala_spark35; sbt "scalafmtCheckAll; scalafixAll --check"; )

check_format_md:
	@echo "Checking markdown files..."
	mdformat --check ${MD_FILES}

check_format: check_format_py check_format_scala check_format_md



# Set PY_TEST_FILES=<TEST_FILE_NAME_GLOB> to test a specifc file.
# Ex. `make integration_test PY_TEST_FILES="dataflow_test.py"`
# By default, runs all tests under python/tests/integration.
# See the help text for "--test_file_pattern" in python/tests/test_args.py for more details.
integration_test:
	( \
	cd python ;\
	python -m tests.integration.main \
	--env=test \
	--resource_config_uri=${GIGL_TEST_DEFAULT_RESOURCE_CONFIG} \
	--test_file_pattern=$(PY_TEST_FILES) \
	)

notebooks_test:
	RESOURCE_CONFIG_PATH=${GIGL_TEST_DEFAULT_RESOURCE_CONFIG} python -m testing.notebooks_test

mock_assets:
	( cd python ; python -m gigl.src.mocking.dataset_asset_mocking_suite --resource_config_uri="deployment/configs/e2e_cicd_resource_config.yaml" --env test)

format_py:
	autoflake --config python/pyproject.toml ${PYTHON_DIRS}
	isort --settings-path=python/pyproject.toml ${PYTHON_DIRS}
	black --config=python/pyproject.toml ${PYTHON_DIRS}

format_scala:
	# We run "clean" before the formatting because otherwise some "scalafix.sbt.ScalafixFailed: NoFilesError" may get thrown after switching branches...
	# TODO(kmonte): Once open sourced, follow up with scalafix team on this.
	( cd scala; sbt clean scalafixAll scalafmtAll )
	( cd scala_spark35; sbt clean scalafixAll scalafmtAll )

format_md:
	@echo "Formatting markdown files..."
	mdformat ${MD_FILES}

format: format_py format_scala format_md


type_check:
	mypy ${PYTHON_DIRS} --check-untyped-defs

# compiles current working state of scala projects to local jars
compile_jars:
	@echo "Compiling jars..."
	@python -m scripts.scala_packager

# Removes local jar files from python/deps directory
remove_jars:
	@echo "Removing jars..."
	rm -rf python/deps/scala/subgraph_sampler/jars/*

push_cpu_docker_image:
	@python -m scripts.build_and_push_docker_image --predefined_type cpu --image_name ${DOCKER_IMAGE_MAIN_CPU_NAME_WITH_TAG}

push_cuda_docker_image:
	@python -m scripts.build_and_push_docker_image --predefined_type cuda --image_name ${DOCKER_IMAGE_MAIN_CUDA_NAME_WITH_TAG}

push_dataflow_docker_image:
	@python -m scripts.build_and_push_docker_image --predefined_type dataflow --image_name ${DOCKER_IMAGE_DATAFLOW_RUNTIME_NAME_WITH_TAG}

push_new_docker_images: push_cuda_docker_image push_cpu_docker_image push_dataflow_docker_image
	# Dockerize the src code and push it to gcr.
	# You will need to update the base image tag below whenever the requirements are updated by:
	#   1) running `make push_new_docker_base_image`
	#   2) Replace the git hash `DOCKER_LATEST_BASE_IMAGE_TAG` that tags the base image with the new generated tag
	# Note: don't forget to `make generate_cpu_hashed_requirements` and `make generate_cuda_hashed_requirements`
	# before running this if you've updated requirements.in
	# You may be able to utilize git comment `/make_cuda_hashed_req` to help you build the cuda hashed req as well
	# See ci.yaml or type in `/help` in your PR for more info.
	@echo "All Docker images compiled and pushed"

push_dev_workbench_docker_image: compile_jars
	@python -m scripts.build_and_push_docker_image --predefined_type=dev_workbench --image_name=${DEFAULT_GIGL_RELEASE_DEV_WORKBENCH_IMAGE}


# Set compiled_pipeline path so compile_gigl_kubeflow_pipeline knows where to save the pipeline to so
# that the e2e test can use it.
run_cora_nalp_e2e_test: compiled_pipeline_path:=GIGL_E2E_TEST_COMPILED_PIPELINE_PATH
run_cora_nalp_e2e_test: compile_gigl_kubeflow_pipeline
run_cora_nalp_e2e_test:
	python python/tests/e2e_tests/e2e_test.py \
		--compiled_pipeline_path=$(compiled_pipeline_path) \
		--test_spec_uri="python/tests/e2e_tests/e2e_tests.yaml" \
		--test_names="cora_nalp_test"

run_cora_snc_e2e_test: compiled_pipeline_path:=GIGL_E2E_TEST_COMPILED_PIPELINE_PATH
run_cora_snc_e2e_test: compile_gigl_kubeflow_pipeline
run_cora_snc_e2e_test:
	python python/tests/e2e_tests/e2e_test.py \
		--compiled_pipeline_path=$(compiled_pipeline_path) \
		--test_spec_uri="python/tests/e2e_tests/e2e_tests.yaml" \
		--test_names="cora_snc_test"

run_cora_udl_e2e_test: compiled_pipeline_path:=GIGL_E2E_TEST_COMPILED_PIPELINE_PATH
run_cora_udl_e2e_test: compile_gigl_kubeflow_pipeline
run_cora_udl_e2e_test:
	python python/tests/e2e_tests/e2e_test.py \
		--compiled_pipeline_path=$(compiled_pipeline_path) \
		--test_spec_uri="python/tests/e2e_tests/e2e_tests.yaml" \
		--test_names="cora_udl_test"

run_dblp_nalp_e2e_test: compiled_pipeline_path:=GIGL_E2E_TEST_COMPILED_PIPELINE_PATH
run_dblp_nalp_e2e_test: compile_gigl_kubeflow_pipeline
run_dblp_nalp_e2e_test:
	python python/tests/e2e_tests/e2e_test.py \
		--compiled_pipeline_path=$(compiled_pipeline_path) \
		--test_spec_uri="python/tests/e2e_tests/e2e_tests.yaml" \
		--test_names="dblp_nalp_test"

run_hom_cora_sup_e2e_test: compiled_pipeline_path:=GIGL_E2E_TEST_COMPILED_PIPELINE_PATH
run_hom_cora_sup_e2e_test: compile_gigl_kubeflow_pipeline
run_hom_cora_sup_e2e_test:
	python python/tests/e2e_tests/e2e_test.py \
		--compiled_pipeline_path=$(compiled_pipeline_path) \
		--test_spec_uri="python/tests/e2e_tests/e2e_tests.yaml" \
		--test_names="hom_cora_sup_test"

run_het_dblp_sup_e2e_test: compiled_pipeline_path:=GIGL_E2E_TEST_COMPILED_PIPELINE_PATH
run_het_dblp_sup_e2e_test: compile_gigl_kubeflow_pipeline
run_het_dblp_sup_e2e_test:
	python python/tests/e2e_tests/e2e_test.py \
		--compiled_pipeline_path=$(compiled_pipeline_path) \
		--test_spec_uri="python/tests/e2e_tests/e2e_tests.yaml" \
		--test_names="het_dblp_sup_test"

run_all_e2e_tests: compiled_pipeline_path:=GIGL_E2E_TEST_COMPILED_PIPELINE_PATH
run_all_e2e_tests: compile_gigl_kubeflow_pipeline
run_all_e2e_tests:
	python python/tests/e2e_tests/e2e_test.py \
		--compiled_pipeline_path=$(compiled_pipeline_path) \
		--test_spec_uri="python/tests/e2e_tests/e2e_tests.yaml"


# Compile an instance of a kfp pipeline
# If you want to compile a pipeline and save it to a specific path, set compiled_pipeline_path
# Example:
# `make compiled_pipeline_path="/tmp/gigl/my_pipeline.yaml" compile_gigl_kubeflow_pipeline`
# Can be a GCS URI as well
compile_gigl_kubeflow_pipeline: compile_jars push_new_docker_images
	python -m gigl.orchestration.kubeflow.runner \
		--action=compile \
		--container_image_cuda=${DOCKER_IMAGE_MAIN_CUDA_NAME_WITH_TAG} \
		--container_image_cpu=${DOCKER_IMAGE_MAIN_CPU_NAME_WITH_TAG} \
		--container_image_dataflow=${DOCKER_IMAGE_DATAFLOW_RUNTIME_NAME_WITH_TAG} \
		$(if $(compiled_pipeline_path),--compiled_pipeline_path=$(compiled_pipeline_path)) \

_skip_build_deps:
	@echo "compiled_pipeline_path was provided. Skipping build dependencies i.e. Compiling jars and pushing docker images"

# Compile and run an instance of pipelines
# Example:
# make \
  job_name="{alias}_run_dev_mag240m_kfp_pipeline" \
  start_at="config_populator" \
  task_config_uri="examples/MAG240M/task_config.yaml" \
  resource_config_uri="examples/MAG240M/resource_config.yaml" \
  run_dev_gnn_kubeflow_pipeline
# If you have precompiled to some specified poth using `make compile_gigl_kubeflow_pipeline`
# You can use it here instead of re-compiling by setting `compiled_pipeline_path`
# Example:
# make \
# 	job_name=... \ , and other params
# 	compiled_pipeline_path="/tmp/gigl/my_pipeline.yaml" \
# 	run_dev_gnn_kubeflow_pipeline
run_dev_gnn_kubeflow_pipeline: $(if $(compiled_pipeline_path), _skip_build_deps, compile_jars push_new_docker_images)
	python -m gigl.orchestration.kubeflow.runner \
		$(if $(compiled_pipeline_path),,--container_image_cuda=${DOCKER_IMAGE_MAIN_CUDA_NAME_WITH_TAG}) \
		$(if $(compiled_pipeline_path),,--container_image_cpu=${DOCKER_IMAGE_MAIN_CPU_NAME_WITH_TAG}) \
		$(if $(compiled_pipeline_path),,--container_image_dataflow=${DOCKER_IMAGE_DATAFLOW_RUNTIME_NAME_WITH_TAG}) \
		--action=$(if $(compiled_pipeline_path),run_no_compile,run) \
		--job_name=$(job_name) \
		--start_at=$(start_at) \
		$(if $(stop_after),--stop_after=$(stop_after)) \
		--task_config_uri=$(task_config_uri) \
		--resource_config_uri=$(resource_config_uri) \
		--pipeline_tag=$(GIT_HASH) \
		$(if $(compiled_pipeline_path),--compiled_pipeline_path=$(compiled_pipeline_path)) \


clean_build_files_py:
	find . -name "*.pyc" -exec rm -f {} \;

clean_build_files_scala:
	( cd scala; sbt clean; find . -type d -name "target" -prune -exec rm -rf {} \; )
	( cd scala_spark35; sbt clean; find . -type d -name "target" -prune -exec rm -rf {} \; )

clean_build_files: clean_build_files_py clean_build_files_scala

# Call to generate new proto definitions if any of the .proto files have been changed.
# We intentionally rebuild *all* protos with one commmand as they should all be in sync.
# Run `make install_dev_deps` to setup the correct protoc versions.
compile_protos:
	tools/python_protoc/bin/protoc \
	--proto_path=proto \
	--python_out=./python \
	--mypy_out=./python \
	proto/snapchat/research/gbml/*.proto

	tools/scalapbc/scalapbc-0.11.11/bin/scalapbc \
		--proto_path=proto \
		--scala_out=scala/common/src/main/scala \
		proto/snapchat/research/gbml/*.proto

	tools/scalapbc/scalapbc-0.11.14/bin/scalapbc \
		--proto_path=proto \
		--scala_out=scala_spark35/common/src/main/scala \
		proto/snapchat/research/gbml/*.proto


spark_run_local_test:
	tools/scala/spark-3.1.3-bin-hadoop3.2/bin/spark-submit \
		--class org.apache.spark.examples.SparkPi \
		--master local[8] \
		tools/scala/spark-3.1.3-bin-hadoop3.2/examples/jars/spark-examples_2.12-3.1.3.jar \
		100

stop_toaster:
	# Stop all existing running docker containers, if no containers to stop continue
	docker stop $(shell docker ps -a -q) || true
	# Deletes everything associated with all stopped containers including dangling resources
	docker system prune -a --volumes
	docker buildx prune

build_docs:
	sphinx-build -M clean . gh_pages_build
	sphinx-build -M html . gh_pages_build

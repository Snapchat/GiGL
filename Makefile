include dep_vars.env

SHELL := /bin/bash
CONDA_ENV_NAME=gnn
PYTHON_VERSION=3.9
PIP_VERSION=25.0.1
DATE:=$(shell /bin/date "+%Y%m%d-%H%M")

# GIT HASH, or empty string if not in a git repo.
GIT_HASH?=$(shell git rev-parse HEAD 2>/dev/null || "")

# You can override GIGL_PROJECT by setting it in your environment i.e.
# adding `export GIGL_PROJECT=your_project` to your shell config (~/.bashrc, ~/.zshrc, etc.)
GIGL_PROJECT?=external-snap-ci-github-gigl
DOCKER_IMAGE_DATAFLOW_RUNTIME_NAME:=us-central1-docker.pkg.dev/${GIGL_PROJECT}/gigl-base-images/src-cpu-dataflow
DOCKER_IMAGE_MAIN_CUDA_NAME:=us-central1-docker.pkg.dev/${GIGL_PROJECT}/gigl-base-images/src-cuda
DOCKER_IMAGE_MAIN_CPU_NAME:=us-central1-docker.pkg.dev/${GIGL_PROJECT}/gigl-base-images/src-cpu

DOCKER_IMAGE_DATAFLOW_RUNTIME_NAME_WITH_TAG:=${DOCKER_IMAGE_DATAFLOW_RUNTIME_NAME}:${DATE}
DOCKER_IMAGE_MAIN_CUDA_NAME_WITH_TAG:=${DOCKER_IMAGE_MAIN_CUDA_NAME}:${DATE}
DOCKER_IMAGE_MAIN_CPU_NAME_WITH_TAG:=${DOCKER_IMAGE_MAIN_CPU_NAME}:${DATE}

PYTHON_DIRS:=examples python shared scripts
PY_TEST_FILES?="*_test.py"

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
	conda create -y -c conda-forge --name ${CONDA_ENV_NAME} python=${PYTHON_VERSION} pip=${PIP_VERSION} pip-tools
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
	--extra torch25-cpu --extra transform \
	./python/pyproject.toml

# Can only be run on an arm64 mac, otherwise generated hashed req file will be wrong
generate_dev_mac_arm64_cpu_hashed_requirements:
	pip-compile -v --allow-unsafe --generate-hashes --no-emit-index-url --resolver=backtracking \
	--output-file=requirements/dev_darwin_arm64_requirements_unified.txt \
	--extra torch25-cpu --extra transform --extra dev \
	./python/pyproject.toml

# Can only be run on linux, otherwise generated hashed req file will be wrong
generate_linux_cpu_hashed_requirements:
	pip-compile -v --allow-unsafe --generate-hashes --no-emit-index-url --resolver=backtracking \
	--output-file=requirements/linux_cpu_requirements_unified.txt \
	--extra torch25-cpu --extra transform \
	./python/pyproject.toml

# Can only be run on linux, otherwise generated hashed req file will be wrong
generate_dev_linux_cpu_hashed_requirements:
	pip-compile -v --allow-unsafe --generate-hashes --no-emit-index-url --resolver=backtracking \
	--output-file=requirements/dev_linux_cpu_requirements_unified.txt \
	--extra torch25-cpu --extra transform --extra dev \
	./python/pyproject.toml

# Can only be run on linux, otherwise generated hashed req file will be wrong
generate_linux_cuda_hashed_requirements:
	pip-compile  -v --allow-unsafe --generate-hashes --no-emit-index-url --resolver=backtracking \
	--output-file=requirements/linux_cuda_requirements_unified.txt \
	--extra torch25-cuda-121 --extra transform \
	./python/pyproject.toml

# Can only be run on linux, otherwise generated hashed req file will be wrong
generate_dev_linux_cuda_hashed_requirements:
	pip-compile -v --allow-unsafe --generate-hashes --no-emit-index-url --resolver=backtracking \
	--output-file=requirements/dev_linux_cuda_requirements_unified.txt \
	--extra torch25-cuda-121 --extra transform --extra dev \
	./python/pyproject.toml

# These are a collection of tests that are run before anything is installed using tools abialable on host.
# May include tests that check the sanity of the repo state i.e. ones that may even cause the failure of
# installation scripts
precondition_tests:
	python shared/tests/dep_vars_check.py


assert_yaml_configs_parse:
	python scripts/assert_yaml_configs_parse.py -d .

# Set PY_TEST_FILES=<TEST_FILE_NAME_GLOB> to test a specifc file.
# Ex. `make unit_test_py PY_TEST_FILES="eval_metrics_test.py"`
# By default, runs all tests under python/testing/unit.
# See the help text for "--test_file_pattern" in python/tests/test_args.py for more details.
unit_test_py: clean_build_files_py type_check
	( cd python ; \
	python -m tests.unit.main \
		--env=test \
		--resource_config_uri ../deployment/configs/unittest_resource_config.yaml \
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
# By default, runs all tests under python/testing/integration.
# See the help text for "--test_file_pattern" in python/tests/test_args.py for more details.
integration_test:
	( \
	cd python ;\
	python -m tests.integration.main \
	--env=test \
	--resource_config_uri ../deployment/configs/unittest_resource_config.yaml \
	--test_file_pattern=$(PY_TEST_FILES) \
	)

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


# Generic make target to run e2e tests. Used by other make targets to run e2e tests.
# See usage w/ run_cora_nalp_e2e_kfp_test, run_cora_snc_e2e_kfp_test, run_cora_udl_e2e_kfp_test
# and run_all_e2e_tests
_run_e2e_kfp_test: compile_jars push_new_docker_images
	$(eval TRIMMED_BRANCH:=$(shell echo "${GIT_BRANCH}" | tr '/' '_' | tr '-' '_' | cut -c 1-20 | tr '[:upper:]' '[:lower:]'))
	$(eval TRIMMED_TIME:=$(shell date +%s | tail -c 6))
	@should_wait_for_job_to_finish=false
	@( \
		set -e; \
		read -a task_config_uris <<< "$(task_config_uris_str)"; \
		read -a resource_config_uris <<< "$(resource_config_uris_str)"; \
		read -a job_name_prefixes_str <<< "$(job_name_prefixes_str)"; \
		read -a should_compile_then_run <<< "$(should_compile_then_run_str)"; \
		if [ $${#task_config_uris[@]} -ne $${#resource_config_uris[@]} ] || [ $${#task_config_uris[@]} -ne $${#job_name_prefixes_str[@]} ] || [ $${#task_config_uris[@]} -ne $${#should_compile_then_run[@]} ]; then \
			echo "Error: Arrays are not of the same length"; \
			echo "  task_config_uris = $${task_config_uris[@]}"; \
			echo "  resource_config_uris = $${resource_config_uris[@]}";\
			echo "  job_name_prefixes_str = $${job_name_prefixes_str[@]}"; \
			echo "  should_compile_then_run = $${should_compile_then_run_str[@]}"; \
			exit 1; \
		fi; \
		for i in $${!task_config_uris[@]}; do \
			job_name="$${job_name_prefixes_str[$$i]}_${TRIMMED_BRANCH}_${TRIMMED_TIME}"; \
			echo "$${job_name} will use compile_then_run = $${should_compile_then_run[$$i]}"; \
			if [ "$${should_compile_then_run[$$i]}" == "true" ]; then \
				compiled_pipeline_path="/tmp/gigl/$${job_name}.yaml"; \
				CMD="python -m gigl.orchestration.kubeflow.runner \
					--container_image_cuda=${DOCKER_IMAGE_MAIN_CUDA_NAME_WITH_TAG} \
					--container_image_cpu=${DOCKER_IMAGE_MAIN_CPU_NAME_WITH_TAG} \
					--container_image_dataflow=${DOCKER_IMAGE_DATAFLOW_RUNTIME_NAME_WITH_TAG} \
					--action=compile \
					--pipeline_tag=$(GIT_HASH) \
					--compiled_pipeline_path='$${compiled_pipeline_path}'"; \
				echo "Compiling pipeline: $$CMD"; \
				eval "$${CMD}"; \
				CMD="python -m gigl.orchestration.kubeflow.runner \
					--action=run_no_compile \
					$(if $(filter ${should_wait_for_job_to_finish},true),--wait,) \
					--job_name='$${job_name}' \
					--start_at='config_populator' \
					--task_config_uri='$${task_config_uris[$$i]}' \
					--resource_config_uri='$${resource_config_uris[$$i]}' \
					--compiled_pipeline_path='$${compiled_pipeline_path}'"; \
				echo "Running from precompiled pipeline: $$CMD"; \
			else \
				CMD="python -m gigl.orchestration.kubeflow.runner \
					--container_image_cuda=${DOCKER_IMAGE_MAIN_CUDA_NAME_WITH_TAG} \
					--container_image_cpu=${DOCKER_IMAGE_MAIN_CPU_NAME_WITH_TAG} \
					--container_image_dataflow=${DOCKER_IMAGE_DATAFLOW_RUNTIME_NAME_WITH_TAG} \
					--action=run \
					$(if $(filter ${should_wait_for_job_to_finish},true),--wait,) \
					--job_name='$${job_name}' \
					--start_at='config_populator' \
					--pipeline_tag=$(GIT_HASH) \
					--task_config_uri='$${task_config_uris[$$i]}' \
					--resource_config_uri='$${resource_config_uris[$$i]}'"; \
				echo "Running: $$CMD"; \
			fi; \
			if [ "$(should_send_job_to_background)" == true ]; then \
				echo "Will run CMD in background..."; \
				eval "$${CMD} &"; \
				pids+=($$!); \
			else \
				eval "$${CMD}"; \
			fi; \
		done; \
		if [ "$(should_send_job_to_background)" == true ]; then \
			echo "Waiting for background jobs to finish..."; \
			for pid in "$${pids[@]}"; do \
				wait "$$pid"; \
			done; \
			echo "All background jobs finished"; \
		fi; \
	)



# cora_nalp_test_on is run by compiling the pipeline first then using the compiled pipeline to run the job
# Vs. all other pipelines right now are run directly i.e. compilation + run together in the same process
run_cora_nalp_e2e_kfp_test: job_name_prefixes_str:="cora_nalp_test_on"
run_cora_nalp_e2e_kfp_test: task_config_uris_str:="gigl/src/mocking/configs/e2e_node_anchor_based_link_prediction_template_gbml_config.yaml"
run_cora_nalp_e2e_kfp_test: resource_config_uris_str:="deployment/configs/e2e_cicd_resource_config.yaml"
run_cora_nalp_e2e_kfp_test: should_compile_then_run_str:="true"
run_cora_nalp_e2e_kfp_test: _run_e2e_kfp_test

run_cora_snc_e2e_kfp_test: job_name_prefixes_str:="cora_snc_test_on"
run_cora_snc_e2e_kfp_test: task_config_uris_str:="gigl/src/mocking/configs/e2e_supervised_node_classification_template_gbml_config.yaml"
run_cora_snc_e2e_kfp_test: resource_config_uris_str:="deployment/configs/e2e_cicd_resource_config.yaml"
run_cora_snc_e2e_kfp_test: should_compile_then_run_str:="false"
run_cora_snc_e2e_kfp_test: _run_e2e_kfp_test

# Note UDL dataset produces a transient issue due to UDL Split Strategy
# where in some cases the root node doesn't properly get added back to
# the returned subgraph. Meaning, trainer will fail.
run_cora_udl_e2e_kfp_test: job_name_prefixes_str:="cora_udl_test_on"
run_cora_udl_e2e_kfp_test: task_config_uris_str:="gigl/src/mocking/configs/e2e_udl_node_anchor_based_link_prediction_template_gbml_config.yaml"
run_cora_udl_e2e_kfp_test: resource_config_uris_str:="deployment/configs/e2e_cicd_resource_config.yaml"
run_cora_udl_e2e_kfp_test: should_compile_then_run_str:="false"
run_cora_udl_e2e_kfp_test: _run_e2e_kfp_test

run_dblp_nalp_e2e_kfp_test: job_name_prefixes_str:="dblp_nalp_test_on"
run_dblp_nalp_e2e_kfp_test: task_config_uris_str:="gigl/src/mocking/configs/dblp_node_anchor_based_link_prediction_template_gbml_config.yaml"
run_dblp_nalp_e2e_kfp_test: resource_config_uris_str:="deployment/configs/e2e_cicd_resource_config.yaml"
run_dblp_nalp_e2e_kfp_test: should_compile_then_run_str:="false"
run_dblp_nalp_e2e_kfp_test: _run_e2e_kfp_test

run_cora_glt_udl_kfp_test: job_name_prefixes_str:="cora_glt_udl_test_on"
run_cora_glt_udl_kfp_test: task_config_uris_str:="examples/distributed/configs/e2e_cora_udl_glt_task_config.yaml"
run_cora_glt_udl_kfp_test: resource_config_uris_str:="deployment/configs/e2e_glt_resource_config.yaml"
run_cora_glt_udl_kfp_test: should_compile_then_run_str:="false"
run_cora_glt_udl_kfp_test: _run_e2e_kfp_test

# Spawns a background job for each e2e test defined by job_name_prefix, task_config_uri, and resource_config_uri
# Waits for all jobs to finish since should_wait_for_job_to_finish:=true
run_all_e2e_tests: should_send_job_to_background:=true
run_all_e2e_tests: should_wait_for_job_to_finish:=true
run_all_e2e_tests: job_name_prefixes_str:=\
		"cora_nalp_test_on" \
		"cora_snc_test_on" \
		"dblp_nalp_test_on" \
		"cora_glt_udl_test_on"
# Removed UDL due to transient issue:
# "gigl/src/mocking/configs/e2e_udl_node_anchor_based_link_prediction_template_gbml_config.yaml"
run_all_e2e_tests: task_config_uris_str:=\
		"gigl/src/mocking/configs/e2e_node_anchor_based_link_prediction_template_gbml_config.yaml" \
		"gigl/src/mocking/configs/e2e_supervised_node_classification_template_gbml_config.yaml" \
		"gigl/src/mocking/configs/dblp_node_anchor_based_link_prediction_template_gbml_config.yaml" \
		"examples/distributed/configs/e2e_cora_udl_glt_task_config.yaml"
run_all_e2e_tests: resource_config_uris_str:=\
		"deployment/configs/e2e_cicd_resource_config.yaml"\
		"deployment/configs/e2e_cicd_resource_config.yaml"\
		"deployment/configs/e2e_cicd_resource_config.yaml"\
		"deployment/configs/e2e_glt_resource_config.yaml"
run_all_e2e_tests: should_compile_then_run_str:=\
		"true" \
		"false" \
		"false" \
		"false"
run_all_e2e_tests: _run_e2e_kfp_test


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

release_gigl:
	@echo "This needs to be implemented"

publish_docs:
	@echo "This needs to be implemented"

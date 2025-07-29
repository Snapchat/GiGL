DATE:=$(shell /bin/date "+%Y%m%d_%H%M")
PROJECT?=external-snap-ci-github-gigl

DOCKER_ARTIFACT_REGISTRY?=us-central1-docker.pkg.dev/${PROJECT}/gigl-base-images

# TODO: Make this work in future PRs; stubbing for now.
run_api_test:
	$(eval version:=$(shell python -c "import gigl; print(gigl.__version__.replace('.', '_'))"))
	$(eval job_name:=api_test_on_${version}_${DATE})
	python -m gigl.orchestration.kubeflow.runner \
		--action=run \
		--job_name=$(job_name) \
		--start_at=config_populator \
		--task_config_uri=api_test_task_config.yaml \
		--resource_config_uri=api_test_resource_config.yaml \
		--extra_source_dir=. \
		--export_docker_artifact_registry=${DOCKER_ARTIFACT_REGISTRY} \
		--wait

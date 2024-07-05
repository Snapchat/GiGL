unit_test:
	@echo "Place holder unit test running..."

integration_test:
	@echo "Place holder integration test running..."

integration_e2e_test:
	@echo "Place holder end to end test running..."

install_dev_deps:
	pip install numpy

generate_dev_linux_cuda_hashed_requirements:
	pip-compile -v --allow-unsafe --generate-hashes --no-emit-index-url --resolver=backtracking \
	--output-file=requirements/dev_linux_cuda_requirements_unified.txt \
	--extra torch21-cuda-118 --extra transform --extra dev \
	./python/pyproject.toml
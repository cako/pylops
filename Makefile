PIP := $(shell command -v pip3 2> /dev/null || command which pip 2> /dev/null)
PYTHON := $(shell command -v python3 2> /dev/null || command which python 2> /dev/null)
UV := $(shell command -v uv 2> /dev/null || command which uv 2> /dev/null)
NOX := $(shell command -v nox 2> /dev/null || command which nox 2> /dev/null)

.PHONY: install_conda dev-install_conda dev-install_conda_intel_mkl dev-install_conda_arm
.PHONY: install_conda dev-install_conda dev-install_conda_intel_mkl dev-install_conda_arm
.PHONY: tests tests_cpu_ongpu tests_gpu tests_uv tests_cpu_ongpu_uv tests_gpu_uv tests_nox
.PHONY: doc doc_uv docupdate docupdate_uv servedoc lint lint_uv typeannot typeannot_uv
.PHONY: coverage coverage_uv

pipcheck:
ifndef PIP
	$(error "Ensure pip or pip3 are in your PATH")
endif
	@echo Using pip: $(PIP)

pythoncheck:
ifndef PYTHON
	$(error "Ensure python or python3 are in your PATH")
endif
	@echo Using python: $(PYTHON)

uvcheck:
ifndef UV
	$(error "Ensure uv is in your PATH")
endif
	@echo Using uv: $(UV)

noxcheck:
ifndef NOX
	$(error "Ensure nox is in your PATH")
endif
	@echo Using nox: $(NOX)

install:
	make pipcheck
	$(PIP) install -r requirements.txt && $(PIP) install .

dev-install:
	make pipcheck
	$(PIP) install -r requirements-dev.txt &&\
	$(PIP) install -r requirements-pyfftw.txt &&\
	$(PIP) install -r requirements-torch.txt && $(PIP) install -e .

dev-install_intel_mkl:
	make pipcheck
	$(PIP) install -r requirements-intel-mkl.txt &&\
	$(PIP) install -r requirements-dev.txt &&\
	$(PIP) install -r requirements-torch.txt && $(PIP) install -e .

dev-install_gpu:
	make pipcheck
	$(PIP) install -r requirements-dev-gpu.txt &&\
	$(PIP) install -e .

install_conda:
	conda env create -f environment.yml && source ${CONDA_PREFIX}/etc/profile.d/conda.sh && conda activate pylops && pip install .

dev-install_conda:
	conda env create -f environment-dev.yml && source ${CONDA_PREFIX}/etc/profile.d/conda.sh && conda activate pylops && pip install -e .

dev-install_conda_intel_mkl:
	conda env create -f environment-dev-intel-mkl.yml && source ${CONDA_PREFIX}/etc/profile.d/conda.sh && conda activate pylops && pip install -e .

dev-install_conda_arm:
	conda env create -f environment-dev-arm.yml && source ${CONDA_PREFIX}/etc/profile.d/conda.sh && conda activate pylops && pip install -e .

dev-install_conda_gpu:
	conda env create -f environment-dev-gpu.yml && source ${CONDA_PREFIX}/etc/profile.d/conda.sh && conda activate pylops_gpu && pip install -e .

dev-install_uv:
	make uvcheck
	$(UV) sync --locked --all-extras --all-groups

tests:
	# Run tests with CPU
	make pythoncheck
	pytest

tests_uv:
	# Run tests with CPU
	make uvcheck
	$(UV) run pytest

tests_nox:
	make noxcheck
	$(NOX) -s tests

tests_cpu_ongpu:
	# Run tests with CPU on a system with GPU (and CuPy installed)
	make pythoncheck
	export CUPY_PYLOPS=0 && export TEST_CUPY_PYLOPS=0 && pytest

tests_cpu_ongpu_uv:
	# Run tests with CPU on a system with GPU (and CuPy installed)
	make pythoncheck
	export CUPY_PYLOPS=0 && export TEST_CUPY_PYLOPS=0 && $(UV) run pytest

tests_gpu:
	# Run tests with GPU (requires CuPy to be installed)
	make pythoncheck
	export TEST_CUPY_PYLOPS=1 && pytest

tests_gpu_uv:
	# Run tests with GPU (requires CuPy to be installed)
	make pythoncheck
	export TEST_CUPY_PYLOPS=1 && $(UV) run pytest

doc:
	cd docs && rm -rf source/api/generated && rm -rf source/gallery &&\
	rm -rf source/tutorials && rm -rf source/examples &&\
	rm -rf build && make html && cd ..

doc_uv:
	make uvcheck
	cd docs  && rm -rf source/api/generated && rm -rf source/gallery &&\
	rm -rf source/tutorials && rm -rf source/examples &&\
	rm -rf build && $(UV) run make html && cd ..

docupdate:
	cd docs && make html && cd ..

docupdate_uv:
	make uvcheck
	cd docs && $(UV) run make html && cd ..

servedoc:
	make pythoncheck
	$(PYTHON) -m http.server --directory docs/build/html/

servedoc_uv:
	make uvcheck
	$(UV) run python -m http.server --directory docs/build/html/

lint:
	ruff check docs/source examples/ pylops/ pytests/ tutorials/

lint_uv:
	make uvcheck
	$(UV) run ruff check docs/source examples/ pylops/ pytests/ tutorials/

typeannot:
	mypy pylops/

typeannot_uv:
	make uvcheck
	$(UV) run mypy pylops/

coverage:
	coverage run --source=pylops -m pytest && \
	coverage xml && coverage html && $(PYTHON) -m http.server --directory htmlcov/

coverage_uv:
	make uvcheck
	$(UV) run coverage run --source=pylops -m pytest  &&\
	$(UV) run coverage xml &&\
	$(UV) run coverage html  &&\
	$(UV) run python -m http.server --directory htmlcov/

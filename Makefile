.PHONY: conda format style types black test link check notebooks
.DEFAULT_GOAL = help

PYTHON = python
VERSION = 3.8
NAME = py_name
ROOT = ./
PIP = pip
CONDA = conda
SHELL = bash
ENV = src
HOST = 127.0.0.1
PORT = 3002

help:	## Display this help
		@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Install Environments

conda:  ## setup a conda environment
		$(info Installing the environment)
		@printf "Creating conda environment...\n"
		${CONDA} create env create -f environment.yml
		@printf "\n\nConda environment created! \033[1;34mRun \`conda activate ${NAME}\` to activate it.\033[0m\n\n\n"

conda_dev:  ## setup a conda environment for development
		@printf "Creating conda dev environment...\n"
		${CONDA} create env create -f environment_dev.yml
		@printf "\n\nConda dev environment created! \033[1;34mRun \`conda activate ${NAME}\` to activate it.\033[0m\n\n\n"

##@ Update Environments

envupdate: ## update conda environment
		@printf "Updating conda environment...\n"
		${CONDA} env update -f environment.yml
		@printf "Conda environment updated!"
	
envupdatedev: ## update conda environment
		@printf "Updating conda dev environment...\n"
		${CONDA} env update -f environment_dev.yml
		@printf "Conda dev environment updated!"

##@ Formatting

black:  ## Format code in-place using black.
		black ${PKGROOT}/ tests/ -l 79 .

format: ## Code styling - black, isort
		black --check --diff ${PKGROOT} tests
		@printf "\033[1;34mBlack passes!\033[0m\n\n"
		isort ${PKGROOT}/ tests/
		@printf "\033[1;34misort passes!\033[0m\n\n"

style:  ## Code lying - pylint
		@printf "Checking code style with flake8...\n"
		flake8 ${PKGROOT}/
		@printf "\033[1;34mPylint passes!\033[0m\n\n"
		@printf "Checking code style with pydocstyle...\n"
		pydocstyle ${PKGROOT}/
		@printf "\033[1;34mpydocstyle passes!\033[0m\n\n"

lint: format style types  ## Lint code using pydocstyle, black, pylint and mypy.
check: lint test  # Both lint and test code. Runs `make lint` followed by `make test`.

##@ Type Checking

types:	## Type checking with mypy
		@printf "Checking code type signatures with mypy...\n"
		python -m mypy ${PKGROOT}/
		@printf "\033[1;34mMypy passes!\033[0m\n\n"

##@ Testing

test:  ## Test code using pytest.
		@printf "\033[1;34mRunning tests with pytest...\033[0m\n\n"
		pytest -v rbig tests
		@printf "\033[1;34mPyTest passes!\033[0m\n\n"

##@ Notebooks

notebooks: notebooks/* # Convert notebooks to html files
		jupyter nbconvert --config nbconfig.py --execute --ExecutePreprocessor.kernel_name="pymc4-dev" --ExecutePreprocessor.timeout=1200
		rm notebooks/*.html


# JUPYTER NOTEBOOKS
notebooks_to_docs: ## Move notebooks to docs notebooks directory
		@printf "\033[1;34mCreating notebook directory...\033[0m\n"
		mkdir -p docs/notebooks
		@printf "\033[1;34mRemoving old notebooks...\033[0m\n"
		rm -rf docs/notebooks/*.ipynb
		@printf "\033[1;34mCopying Notebooks to directory...\033[0m\n"
		rsync -zarv --progress notebooks/ docs/notebooks/ --include="*.ipynb" --exclude="*.csv" --exclude=".ipynb_checkpoints/" 
		@printf "\033[1;34mDone!\033[0m\n"
jlab_html:
		mkdir -p docs/notebooks
		jupyter nbconvert notebooks/*.ipynb --to html --output-dir docs/notebooks/

docs-live: ## Build mkdocs documentation live
		@printf "\033[1;34mStarting live docs with mkdocs...\033[0m\n"
		mkdocs serve --dev-addr $(HOST):$(PORT) --theme material

docs-live-d:  ## notebooks_to_docs Build mkdocs documentation live (quicker reload)
		@printf "\033[1;34mStarting live docs with mkdocs...\033[0m\n"
		mkdocs serve --dev-addr $(HOST):$(PORT) --dirtyreload --theme material

docs-deploy-all: notebooks_to_docs pdocs ## Deploy docs
		@printf "\033[1;34mDeploying docs...\033[0m\n"
		mkdocs gh-deploy
		@printf "\033[1;34mSuccess...\033[0m\n"

docs-deploy: ## Deploy docs
		@printf "\033[1;34mDeploying docs...\033[0m\n"
		mkdocs gh-deploy
		@printf "\033[1;34mSuccess...\033[0m\n"
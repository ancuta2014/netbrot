PYTHON?=python -X dev

all: help

help: 			## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

format: 					## Run all formatting scripts
.PHONY: format

lint: codespell reuse 			## Run linting checks
.PHONY: lint

codespell:		## Run codespell checks over the documentation
	@codespell --summary \
		--uri-ignore-words-list '*' \
		--ignore-words .codespell-ignore \
		src README.rst
	@echo -e "\e[1;32mcodespell clean!\e[0m"
.PHONY: codespell

reuse:			## Check REUSE license compliance
	$(PYTHON) -m reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"
.PHONY: reuse

PYTHON?=python -X dev

all: help

help: 			## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

# {{{ formatting

format: rustfmt rufffmt	shfmt		## Run all formatting scripts
.PHONY: format

rustfmt:						## Run rustfmt
	cargo fmt -- src/*.rs
	@echo -e "\e[1;32mrustfmt clean!\e[0m"
.PHONY: rustfmt

rufffmt:						## Run ruff format
	ruff format scripts
	ruff check --fix --select=I scripts
	ruff check --fix --select=RUF022 scripts
	@echo -e "\e[1;32mruff format clean!\e[0m"

shfmt:							## Run shfmt format
	shfmt --write --language-dialect bash --indent 4 scripts/*.sh
	@echo -e "\e[1;32mshfmt clean!\e[0m"

# }}}

# {{{ linting

lint: typos reuse ruff clippy	## Run linting checks
.PHONY: lint

typos:			## Run typos over the source code and documentation
	typos --sort
	@echo -e "\e[1;32mtypos clean!\e[0m"
.PHONY: typos

reuse:			## Check REUSE license compliance
	$(PYTHON) -m reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"
.PHONY: reuse

ruff:			## Run ruff checks over the source code
	ruff check scripts
	@echo -e "\e[1;32mruff clean!\e[0m"
.PHONY: ruff

clippy:			## Run clippy lint checks
	cargo clippy --all-targets --all-features
	@echo -e "\e[1;32mclippy clean!\e[0m"

# }}}

# {{{ building

build:			## Build the project in debug mode
	cargo build --locked --all-features --verbose
.PHONY: build

release:		## Build project in release mode
	cargo build --locked --all-features --release
.PHONY: release

# }}}

# {{{ gallery

mat-default:			## Generate default test matrices
	$(PYTHON) scripts/generate-default-exhibits.py \
		--overwrite \
		--outfile data/defaults.npz
.PHONY: mat-default

mat-structural:			## Convert example structural data to npz
	$(PYTHON) scripts/convert-mat.py \
		--overwrite \
		-n 'Structural_Conn' \
		--upper-left='-3.75+2.5j' \
		--lower-right='1.25-2.5j' \
		--outfile data/structural.npz \
		data/structural_scaled_by_task_max.mat
.PHONY: mat-structural

gallery-default:		## Generate default gallery
	$(PYTHON) scripts/generate-matrix-gallery.py \
		--overwrite \
		--outfile src/gallery.rs \
		data/defaults.npz

# }}}

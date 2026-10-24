.PHONY: install test demo doctor bootstrap figures package

install:
	pip install -e .

test:
	pytest -q

demo:
	python scripts/run_toy_demo.py

doctor:
	python scripts/doctor.py

bootstrap:
	bash scripts/bootstrap_third_party.sh

figures:
	python scripts/build_figure_assets.py
	python experiments/make_figures_v6.py

package:
	git archive --format=zip -o PhyGuard-FM-$$(git describe --tags --always).zip HEAD

build:
	jupyter nbconvert Turing.ipynb --to script --output app
	python postprocess.py
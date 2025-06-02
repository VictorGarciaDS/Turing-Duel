build:
	jupyter nbconvert Turing.ipynb --to python --output app
	python postprocess.py
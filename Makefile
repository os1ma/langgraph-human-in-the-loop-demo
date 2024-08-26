.PHONY: all
all: requirements.txt

requirements.txt: poetry.lock
	poetry export --format requirements.txt --output $@

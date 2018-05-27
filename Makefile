# Shorthand to run the examples
SRC_DIR  := $(CURDIR)/examples
EXAMPLES := $(patsubst $(SRC_DIR)/%.py, %, $(wildcard $(SRC_DIR)/*.py))


build:
	bash -c "pip install -r $(CURDIR)/requirements.txt"
	bash -c ". ~/.spinnaker_env && $(MAKE) -C $(CURDIR)/c_models"

$(EXAMPLES): build
	bash -c "export PYTHONPATH=$(CURDIR) && python $(SRC_DIR)/$@.py --show-in --show-out"

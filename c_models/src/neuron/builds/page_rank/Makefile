APP = $(notdir $(CURDIR))
#SPYNNAKER_DEBUG = DEBUG
BUILD_DIR = build/


# Maintains the state of a neuron
NEURON_MODEL = $(EXTRA_SRC_DIR)/neuron/models/neuron_model_page_rank.c
NEURON_MODEL_H = $(EXTRA_SRC_DIR)/neuron/models/neuron_model_page_rank.h

# When to send a spike
THRESHOLD_TYPE_H = $(EXTRA_SRC_DIR)/neuron/threshold_types/threshold_type_page_rank.h

# No-op synapse shaping type
SYNAPSE_TYPE_H = $(EXTRA_SRC_DIR)/neuron/synapse_types/synapse_types_noop.h

# Override defaults from sPyNNaker/neural_modelling/src/neuron
NEURON_C = $(EXTRA_SRC_DIR)/neuron/neuron.c
SPIKE_PROCESSING_C = $(EXTRA_SRC_DIR)/neuron/spike_processing.c


include ../Makefile.common
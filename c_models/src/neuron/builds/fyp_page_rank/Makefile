APP = $(notdir $(CURDIR))
#SPYNNAKER_DEBUG = DEBUG
BUILD_DIR = build/


NEURON_MODEL = $(EXTRA_SRC_DIR)/neuron/models/neuron_model_page_rank.c
NEURON_MODEL_H = $(EXTRA_SRC_DIR)/neuron/models/neuron_model_page_rank.h

# This is the (optional) header containing the additional input type
# ADDITIONAL_INPUT_H = $(EXTRA_SRC_DIR)/neuron/additional_inputs/my_additional_input.h

# Header containing the input type (current in this case)
INPUT_TYPE_H = $(SOURCE_DIR)/neuron/input_types/input_type_current.h

# Header containing the threshold type (static in this case)
THRESHOLD_TYPE_H = $(SOURCE_DIR)/neuron/threshold_types/threshold_type_static.h

# Header containing the synapse shaping type (exponential in this case)
SYNAPSE_TYPE_H = $(SOURCE_DIR)/neuron/synapse_types/synapse_types_exponential_impl.h

# Synapse dynamics type (in this case static i.e. no synapse dynamics)
SYNAPSE_DYNAMICS = $(SOURCE_DIR)/neuron/plasticity/synapse_dynamics_static_impl.c

# Override defaults from sPyNNaker/neural_modelling/src/neuron
NEURON_C = $(EXTRA_SRC_DIR)/neuron/neuron.c
SPIKE_PROCESSING_C = $(EXTRA_SRC_DIR)/neuron/spike_processing.c


include ../Makefile.common
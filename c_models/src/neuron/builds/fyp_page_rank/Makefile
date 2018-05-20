APP = $(notdir $(CURDIR))
BUILD_DIR = build/

# Override defaults
NEURON_C = $(EXTRA_SRC_DIR)/neuron/neuron.c
SPIKE_PROCESSING_C = $(EXTRA_SRC_DIR)/neuron/spike_processing.c

# This is the neuron model implementation
NEURON_MODEL = $(EXTRA_SRC_DIR)/neuron/models/my_neuron_model_impl.c
# This is the header of the neuron model, containing the definition of neuron_t
NEURON_MODEL_H = $(EXTRA_SRC_DIR)/neuron/models/my_neuron_model_impl.h

# This is the (optional) header containing the additional input type
# ADDITIONAL_INPUT_H = $(EXTRA_SRC_DIR)/neuron/additional_inputs/my_additional_input.h

# This is the header containing the input type (current in this case)
INPUT_TYPE_H = $(SOURCE_DIR)/neuron/input_types/input_type_current.h

# This is the header containing the threshold type (static in this case)
THRESHOLD_TYPE_H = $(SOURCE_DIR)/neuron/threshold_types/threshold_type_static.h

# This is the header containing the synapse shaping type (exponential in this case)
SYNAPSE_TYPE_H = $(SOURCE_DIR)/neuron/synapse_types/synapse_types_exponential_impl.h
# This is the synapse dynamics type (in this case static i.e. no synapse dynamics)
SYNAPSE_DYNAMICS = $(SOURCE_DIR)/neuron/plasticity/synapse_dynamics_static_impl.c


# This includes the common Makefile that hides away the details of the build
include ../Makefile.common
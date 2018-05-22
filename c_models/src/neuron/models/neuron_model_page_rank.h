#ifndef _NEURON_MODEL_MY_IMPL_H_
#define _NEURON_MODEL_MY_IMPL_H_

#include <neuron/models/neuron_model.h>

typedef struct neuron_t {

    // TODO: Parameters - make sure these match with the python code,
    // including the order of the variables when returned by
    // get_neural_parameters.

    // The current rank of the neuron
    REAL rank;

    // Pending neuron update: the accumulated / count of ranks received.
    REAL curr_rank_acc;
    REAL curr_rank_count;

} neuron_t;

typedef struct global_neuron_params_t {
    uint32_t machine_time_step;

} global_neuron_params_t;

#endif // _NEURON_MODEL_MY_IMPL_H_


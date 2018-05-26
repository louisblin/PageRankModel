#ifndef _NEURON_MODEL_MY_IMPL_H_
#define _NEURON_MODEL_MY_IMPL_H_

#include <neuron/models/neuron_model.h>

typedef struct neuron_t {

    // Number of edges inbound / leaving that neuron
    uint32_t incoming_edges_count;
    uint32_t outgoing_edges_count;

    // The current rank of the neuron
    REAL rank;

    // Pending neuron update: the accumulated / count of ranks received.
    REAL curr_rank_acc;
    uint32_t curr_rank_count;
    uint32_t has_completed_iter;

} neuron_t;

typedef struct global_neuron_params_t {
    uint32_t machine_time_step;

} global_neuron_params_t;

#endif // _NEURON_MODEL_MY_IMPL_H_


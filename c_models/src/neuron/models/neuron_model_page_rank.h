#ifndef _NEURON_MODEL_PAGE_RANK_H_
#define _NEURON_MODEL_PAGE_RANK_H_

#include <neuron/models/neuron_model.h>
#include <common/maths-util.h>

#define K(n) (n >> 17)

typedef struct neuron_t {

    // Number of edges inbound / leaving that neuron
    uint32_t incoming_edges_count;
    uint32_t outgoing_edges_count;

    // The current rank of the neuron
    UFRACT rank;

    // Pending neuron update: the accumulated / count of ranks received.
    UFRACT curr_rank_acc;
    uint32_t curr_rank_count;
    uint32_t iter_state;

} neuron_t;

typedef struct global_neuron_params_t {
    uint32_t machine_time_step;

} global_neuron_params_t;


void neuron_model_receive_packet(input_t key, spike_t payload, neuron_pointer_t neuron);

REAL neuron_model_get_rank_as_real(neuron_pointer_t neuron);
payload_t neuron_model_get_broadcast_rank(neuron_pointer_t neuron);

bool neuron_model_should_send_pkt(neuron_pointer_t neuron);
void neuron_model_will_send_pkt(neuron_pointer_t neuron);

void neuron_model_iteration_did_finish(neuron_pointer_t neuron);


#endif // _NEURON_MODEL_PAGE_RANK_H_


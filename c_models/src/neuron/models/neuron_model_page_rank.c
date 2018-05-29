#include "neuron_model_page_rank.h"

#include <stdfix.h>
#include <debug.h>

static global_neuron_params_pointer_t global_params;

void neuron_model_set_global_neuron_params(global_neuron_params_pointer_t params) {
    global_params = params;
}

// Triggered when a packet is received
//   or, when all params are UNUSED, to get the number of inputs received
state_t neuron_model_state_update(input_t key, input_t payload, input_t unused,
        neuron_pointer_t neuron) {
    use(unused);

    // Decode key / payload
    index_t idx = (index_t) key;

    union payloadDeserializer {
        uint32_t asInt;
        REAL asReal;
    };
    union payloadDeserializer contrib = { payload };

    // User signals a packet has arrived
    REAL prev_rank_acc = neuron->curr_rank_acc;
    uint32_t prev_rank_count = neuron->curr_rank_count;

    // Saved
    neuron->curr_rank_acc   += contrib.asReal;
    neuron->curr_rank_count += 1;

    log_info("[idx=%03u] neuron_model_state_update: %2.4k/%d + %2.4k = %2.4k/%d [exp=%d]", idx,
        prev_rank_acc, prev_rank_count, contrib.asReal, neuron->curr_rank_acc,
        neuron->curr_rank_count, neuron->incoming_edges_count);

    if ( neuron->curr_rank_count >= neuron->incoming_edges_count ) {
//        neuron->rank = neuron->curr_rank_acc;
//        neuron->curr_rank_acc   = 0;
//        neuron->curr_rank_count = 0;
        neuron->has_completed_iter = 1;
        log_info("[idx=%03u] neuron_model_state_update: iteration completed (%2.4k)", idx,
            neuron->curr_rank_acc);
    }

    return neuron_model_get_membrane_voltage(neuron);
}

// Membrane voltage is defined as the rank here
state_t neuron_model_get_membrane_voltage(neuron_pointer_t neuron) {
    // Check we don't divide by 0
    if (neuron->outgoing_edges_count == 0) {
        return neuron->rank;
    }
    return neuron->rank / neuron->outgoing_edges_count;
}

// Perform operations required to reset the state after a spike
void neuron_model_has_spiked(neuron_pointer_t neuron) {
    log_info("Neuron spiked: rank = %2.4k", neuron->rank);

    // If not expected to receive any packets, iteration is finished for the node
    if (neuron->incoming_edges_count == 0) {
        neuron->has_completed_iter = 1;
    }
}

void neuron_model_print_state_variables(restrict neuron_pointer_t neuron) {
    log_info("rank            = %4.4k", neuron->rank);
    log_info("curr_rank_acc   = %4.4k", neuron->curr_rank_acc);
    log_info("curr_rank_count = %d", neuron->curr_rank_count);
    log_info("has_completed_iter = %d", neuron->has_completed_iter);
}

void neuron_model_print_parameters(restrict neuron_pointer_t neuron) {
    log_info("incoming_edges_count = %d", neuron->incoming_edges_count);
    log_info("outgoing_edges_count = %d", neuron->outgoing_edges_count);
}

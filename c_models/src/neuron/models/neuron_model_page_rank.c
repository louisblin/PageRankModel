#include "neuron_model_page_rank.h"

#include <common/maths-util.h>
#include <debug.h>

#define K(n) (n >> 17)

static global_neuron_params_pointer_t global_params;

typedef enum neuron_model_iteration_state {
    HAS_SENT_PKT, HAS_RECEIVED_PKTS
} neuron_model_iteration_state;


void neuron_model_set_global_neuron_params(global_neuron_params_pointer_t params) {
    global_params = params;
}

// Triggered when a packet is received
//   or, when all params are UNUSED, to get the number of inputs received
void neuron_model_receive_packet(input_t key, spike_t payload, neuron_pointer_t neuron) {
    // Decode key / payload
    index_t idx = (index_t) key;

    union payloadDeserializer {
        spike_t asSpikeT;
        UFRACT asFract;
    };
    union payloadDeserializer contrib = { payload };

    // User signals a packet has arrived
    UFRACT prev_rank_acc = neuron->curr_rank_acc;
    uint32_t prev_rank_count = neuron->curr_rank_count;

    // Saved
    neuron->curr_rank_acc   += contrib.asFract;
    neuron->curr_rank_count += 1;

    log_info("[idx=%03u] neuron_model_state_update: %k/%d + %k = %k/%d [exp=%d]", idx,
        K(prev_rank_acc), prev_rank_count, K(contrib.asFract), K(neuron->curr_rank_acc),
        neuron->curr_rank_count, neuron->incoming_edges_count);

    if ( neuron->curr_rank_count >= neuron->incoming_edges_count ) {
        neuron->has_completed_iter = 1;
        log_info("[idx=%03u] neuron_model_state_update: iteration completed (%k)", idx,
            K(neuron->curr_rank_acc));
    }
}

payload_t neuron_model_get_broadcast_rank(neuron_pointer_t neuron) {
    union payloadSerializer {
        UFRACT asFract;
        payload_t asPayloadT;
    };
    union payloadSerializer rank = { neuron->rank };

    // Check we don't divide by 0
    if (neuron->outgoing_edges_count > 0) {
        rank.asFract /= neuron->outgoing_edges_count;
    }
    return rank.asPayloadT;
}

REAL neuron_model_get_rank_as_real(neuron_pointer_t neuron) {
    union payloadSerializer {
        UFRACT asFract;
        REAL asReal;
    };
    union payloadSerializer rank = { neuron->rank };
    return rank.asReal;
}


// Perform operations required to reset the state after a spike
void neuron_model_will_send_pkt(neuron_pointer_t neuron) {
    log_debug("Neuron spiked: rank = %k[0x%08x]", K(neuron->rank), neuron->rank);

    // If not expected to receive any packets, iteration is finished for the node
    if (neuron->incoming_edges_count == 0) {
        neuron->has_completed_iter = 1;
    }
}

bool neuron_model_has_finished_iteration(neuron_pointer_t neuron) {
    return neuron->has_completed_iter;
}

void neuron_model_iteration_did_finish(neuron_pointer_t neuron) {
    neuron->rank = neuron->curr_rank_acc;
    neuron->curr_rank_acc = 0;
    neuron->curr_rank_count = 0;
    neuron->has_completed_iter = 0;
}

void neuron_model_print_state_variables(restrict neuron_pointer_t neuron) {
    log_info("rank               = %k", K(neuron->rank));
    log_info("curr_rank_acc      = %k", K(neuron->curr_rank_acc));
    log_info("curr_rank_count    = %d", neuron->curr_rank_count);
    log_info("has_completed_iter = %d", neuron->has_completed_iter);
}

void neuron_model_print_parameters(restrict neuron_pointer_t neuron) {
    log_info("incoming_edges_count = %d", neuron->incoming_edges_count);
    log_info("outgoing_edges_count = %d", neuron->outgoing_edges_count);
}

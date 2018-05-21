#include "neuron_model_page_rank.h"

#include <stdfix.h>
#include <debug.h>

static global_neuron_params_pointer_t global_params;

void neuron_model_set_global_neuron_params(global_neuron_params_pointer_t params) {
    global_params = params;
}

// Triggered when a packet is received
//   or, when all params are UNUSED, to get the number of inputs received
state_t neuron_model_state_update(input_t _key, input_t _payload, input_t unused,
        neuron_pointer_t neuron) {
    use(unused);

    // User wants to get the number of incoming edges received since last spike
    if (!_key && !_payload) {
        return neuron->curr_rank_count;
    }

    // User signals a packet has arrived
    index_t key = (key_t) _key;
    REAL contrib = (REAL) _payload;

    log_debug("neuron_model_state_update: %04x=%04x => %03u=%3.3k", key, contrib, key, contrib);

    neuron->curr_rank_acc += contrib;
    neuron->curr_rank_count += REAL_CONST(1);

    log_debug("curr_rank_acc=%3.3k [%04x]  ||  curr_rank_count=%3.3k [%04x]", neuron->curr_rank_acc,
            neuron->curr_rank_acc, neuron->curr_rank_count, neuron->curr_rank_count);

    return neuron->rank;
}

// Membrane voltage is defined as the rank here
state_t neuron_model_get_membrane_voltage(neuron_pointer_t neuron) {
    return neuron->rank;
}

// Perform operations required to reset the state after a spike
void neuron_model_has_spiked(neuron_pointer_t neuron) {

    if (neuron->curr_rank_count == 0) {
        log_info("Neuron spiked without received any packet");
    } else {
        neuron->rank = neuron->curr_rank_acc / neuron->curr_rank_count;
        neuron->curr_rank_acc   = 0;
        neuron->curr_rank_count = 0;
    }
}

void neuron_model_print_state_variables(restrict neuron_pointer_t neuron) {
    log_debug("rank             = %11.4k", neuron->rank);
    log_debug("curr_rank_acc    = %11.4k", neuron->curr_rank_acc);
    log_debug("curr_rank_count  = %11.4k", neuron->curr_rank_count);
}

void neuron_model_print_parameters(restrict neuron_pointer_t neuron) {
    use(neuron);
    log_debug("*** no parameters ***");
}

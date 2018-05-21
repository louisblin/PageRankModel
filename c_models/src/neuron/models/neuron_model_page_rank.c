#include "neuron_model_page_rank.h"

#include <stdfix.h>
#include <debug.h>

static global_neuron_params_pointer_t global_params;

void neuron_model_set_global_neuron_params(
        global_neuron_params_pointer_t params) {

    global_params = params;
}

// Rank does not change over time
state_t neuron_model_state_update(input_t exc_input, input_t inh_input, input_t external_bias,
        neuron_pointer_t neuron) {

    log_info("neuron_model_state_update");
    return neuron->rank;
}

// Membrane voltage is defined as the rank here
state_t neuron_model_get_membrane_voltage(neuron_pointer_t neuron) {
    log_info("neuron_model_get_membrane_voltage %11.4k / %11.4k / %11.4k", neuron->rank,
            neuron->curr_rank_acc, neuron->curr_rank_count);
    return neuron->rank;
}

// Perform operations required to reset the state after a spike
void neuron_model_has_spiked(neuron_pointer_t neuron) {
    log_info("neuron_model_has_spiked %11.4k / %11.4k / %11.4k", neuron->rank,
            neuron->curr_rank_acc, neuron->curr_rank_count);

    if (neuron->curr_rank_count == REAL_CONST( 0.0 )) {
        log_info("Neuron spiked without received any packet");
    } else {
        neuron->rank = neuron->curr_rank_acc / neuron->curr_rank_count;
        neuron->curr_rank_acc   = REAL_CONST( 0.0 );
        neuron->curr_rank_count = REAL_CONST( 0.0 );
    }
}

void neuron_model_print_state_variables(restrict neuron_pointer_t neuron) {
    // The current rank of the neuron
    log_debug("rank             = %11.4k", neuron->rank);

    // Pending neuron update: the accumulated / count of ranks received.
    log_debug("curr_rank_acc    = %11.4k", neuron->curr_rank_acc);
    log_debug("curr_rank_count  = %11.4k", neuron->curr_rank_count);
}

void neuron_model_print_parameters(restrict neuron_pointer_t neuron) {
    log_debug("*** no parameters *** (see neuron_model_print_state_variables)");
}

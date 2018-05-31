/*! \file
 *
 * \brief implementation of the neuron.h interface.
 *
 */

#include "neuron.h"
#include "models/neuron_model_page_rank.h"
#include <neuron/synapse_types/synapse_types.h>
#include <neuron/plasticity/synapse_dynamics.h>
#include <common/out_spikes.h>
#include <common/maths-util.h>
#include <recording.h>
#include <debug.h>
#include <string.h>

// declare spin1_wfi
void spin1_wfi();

#define SPIKE_RECORDING_CHANNEL 0
#define RANK_RECORDING_CHANNEL 1

//! Array of neuron states
static neuron_pointer_t neuron_array;

//! Global parameters for the neurons
static global_neuron_params_pointer_t global_parameters;

//! The key to be used for this core (will be ORed with neuron id)
static key_t key;

//! A checker that says if this model should be transmitting. If set to false
//! by the data region, then this model should not have a key.
static bool use_key;

//! The number of neurons on the core
static uint32_t n_neurons;

//! Keeps track of which neuron have sent their packet during the iteration
static bool *has_sent_packets;

//! The recording flags
static uint32_t recording_flags;

// The synapse shaping parameters
static synapse_param_t *neuron_synapse_shaping_params;

//! storage for neuron state with timestamp
static timed_state_t *ranks;
uint32_t ranks_size;

//! The number of clock ticks to back off before starting the timer, in an attempt to avoid
//!   overloading the network
static uint32_t random_back_off;

//! The number of clock ticks between sending each spike
static uint32_t time_between_spikes;

//! The expected current clock tick of timer_1 when the next spike can be sent
static uint32_t expected_time;

//! The number of recordings outstanding
static uint32_t n_recordings_outstanding = 0;

//! parameters that reside in the neuron_parameter_data_region in human
//! readable form
typedef enum parameters_in_neuron_parameter_data_region {
    RANDOM_BACK_OFF, TIME_BETWEEN_SPIKES, HAS_KEY, TRANSMISSION_KEY,
    N_NEURONS_TO_SIMULATE, INCOMING_SPIKE_BUFFER_SIZE,
    START_OF_GLOBAL_PARAMETERS,
} parameters_in_neuron_parameter_data_region;


//! private method for doing output debug data on the neurons
static inline void _print_neurons() {

//! only if the models are compiled in debug mode will this method contain these lines.
//#if LOG_LEVEL >= LOG_DEBUG
    log_info("-------------------------------------\n");
    for (index_t n = 0; n < n_neurons; n++) {
        neuron_model_print_state_variables(&(neuron_array[n]));
    }
    log_info("-------------------------------------\n");
    //}
//#endif // LOG_LEVEL >= LOG_DEBUG
}

//! private method for doing output debug data on the neurons
static inline void _print_neuron_parameters() {

//! only if the models are compiled in debug mode will this method contain these lines.
//#if LOG_LEVEL >= LOG_DEBUG
    log_info("-------------------------------------\n");
    for (index_t n = 0; n < n_neurons; n++) {
        neuron_model_print_parameters(&(neuron_array[n]));
    }
    log_info("-------------------------------------\n");
    //}
//#endif // LOG_LEVEL >= LOG_DEBUG
}

//! \brief sets to false the has_sent_packets array at the start of a new iteration
static inline void _reset_has_sent_packets() {
    log_info("RESETTING has_sent_packets");
    for (index_t neuron_index = 0; neuron_index < n_neurons; neuron_index++) {
        has_sent_packets[neuron_index] = false;
    }
}

//! \brief does the memory copy for the neuron parameters
//! \param[in] address: the address where the neuron parameters are stored
//! in SDRAM
//! \return bool which is true if the mem copy's worked, false otherwise
bool _neuron_load_neuron_parameters(address_t address){
    uint32_t next = START_OF_GLOBAL_PARAMETERS;

    log_info("loading neuron global parameters");
    memcpy(global_parameters, &address[next], sizeof(global_neuron_params_t));
    next += sizeof(global_neuron_params_t) / 4;

    log_info("loading neuron local parameters");
    memcpy(neuron_array, &address[next], n_neurons * sizeof(neuron_t));

    neuron_model_set_global_neuron_params(global_parameters);

    return true;
}

//! \brief interface for reloading neuron parameters as needed
//! \param[in] address: the address where the neuron parameters are stored in SDRAM
//! \return bool which is true if the reload of the neuron parameters was
//! successful or not
bool neuron_reload_neuron_parameters(address_t address) {
    log_info("neuron_reloading_neuron_parameters: starting");
    if (!_neuron_load_neuron_parameters(address)){
        return false;
    }

    // for debug purposes, print the neuron parameters
    _print_neuron_parameters();
    return true;
}

//! \brief Set up the neuron models
//! \param[in] address the absolute address in SDRAM for the start of the NEURON_PARAMS data region
//!            in SDRAM
//! \param[in] recording_flags_param the recordings parameters (contains which regions are active
//!            and how big they are)
//! \param[out] n_neurons_value The number of neurons this model is to emulate
//! \return true if the initialisation was successful, otherwise false
bool neuron_initialise(address_t address, uint32_t recording_flags_param,
        uint32_t *n_neurons_value, uint32_t *incoming_spike_buffer_size) {
    log_info("neuron_initialise: starting");

    random_back_off     = address[RANDOM_BACK_OFF];
    time_between_spikes = address[TIME_BETWEEN_SPIKES] * sv->cpu_clk;
    log_info("\t back off = %u, time between spikes %u", random_back_off, time_between_spikes);

    // Check if there is a key to use
    use_key = address[HAS_KEY];

    // Read the spike key to use
    key = address[TRANSMISSION_KEY];

    // log if this model is expecting to transmit
    if (!use_key) {
        log_info("\tThis model is not expecting to transmit as it has no key");
    } else {
        log_info("\tThis model is expected to transmit with key = %08x", key);
    }

    // Read the neuron details
    n_neurons = address[N_NEURONS_TO_SIMULATE];
    *n_neurons_value = n_neurons;

    // Read the size of the incoming spike buffer to use
    *incoming_spike_buffer_size = address[INCOMING_SPIKE_BUFFER_SIZE];

    // log message for debug purposes
    log_info("\t neurons = %u, spike buffer size = %u, params size = %u",
        n_neurons, *incoming_spike_buffer_size, sizeof(neuron_t));

    // Allocate DTCM for the global parameter details
    if (sizeof(global_neuron_params_t) > 0) {
        global_parameters = (global_neuron_params_t *) spin1_malloc(sizeof(global_neuron_params_t));
        if (global_parameters == NULL) {
            log_error("Unable to allocate global neuron parameters - Out of DTCM");
            return false;
        }
    }

    // Allocate DTCM for neuron array
    if (sizeof(neuron_t) != 0) {
        neuron_array = (neuron_t *) spin1_malloc(n_neurons * sizeof(neuron_t));
        if (neuron_array == NULL) {
            log_error("Unable to allocate neuron array - Out of DTCM");
            return false;
        }
    }

    // Allocate DTCM for has_sent_packet
    if (n_neurons > 0) {
        has_sent_packets = (bool *) spin1_malloc(sizeof(bool) * n_neurons);
        if (has_sent_packets == NULL) {
            log_error("Unable to allocate has_sent_packets - Out of DTCM");
            return false;
        }
        _reset_has_sent_packets();
    }

    // Load the data into the allocated DTCM spaces.
    if (!_neuron_load_neuron_parameters(address)){
        return false;
    }

    // Set up the out spikes array
    if (!out_spikes_initialize(n_neurons)) {
        return false;
    }

    recording_flags = recording_flags_param;

    ranks_size = sizeof(uint32_t) + sizeof(state_t) * n_neurons;
    ranks = (timed_state_t *) spin1_malloc(ranks_size);

    _print_neuron_parameters();

    return true;
}


//! \brief stores neuron parameter back into sdram
//! \param[in] address: the address in sdram to start the store
void neuron_store_neuron_parameters(address_t address){

    uint32_t next = START_OF_GLOBAL_PARAMETERS;

    log_info("writing neuron global parameters");
    memcpy(&address[next], global_parameters, sizeof(global_neuron_params_t));
    next += sizeof(global_neuron_params_t) / 4;

    log_info("writing neuron local parameters");
    memcpy(&address[next], neuron_array, n_neurons * sizeof(neuron_t));
}

//! \setter for the internal input buffers
//! \param[in] input_buffers_value the new input buffers
void neuron_set_neuron_synapse_shaping_params(synapse_param_t *value){
    neuron_synapse_shaping_params = value;
}

void recording_done_callback() {
    n_recordings_outstanding -= 1;
}

//! \executes all the updates to neural parameters when a given timer period has occurred.
//! \param[in] time the timer tick  value currently being executed
void neuron_do_timestep_update(timer_t time) {

    log_info("\n\n===== TIME STEP = %u =====", time);

    // Track sent packets for each node
    bool hasSentAllPackets = true;

    // Wait a random number of clock cycles
    uint32_t random_back_off_time = tc[T1_COUNT] - random_back_off;
    while (tc[T1_COUNT] > random_back_off_time) {
        // Do Nothing
    }

    // Set the next expected time to wait for between spike sending
    expected_time = tc[T1_COUNT] - time_between_spikes;

    // Wait until recordings have completed, to ensure the recording space can be re-written
    while (n_recordings_outstanding > 0) {
        spin1_wfi();
    }

    // Reset the out spikes before starting
    out_spikes_reset();

    // update each neuron individually
    for (index_t neuron_index = 0; neuron_index < n_neurons; neuron_index++) {
        // Get the parameters for this neuron
        neuron_pointer_t neuron = &neuron_array[neuron_index];

        // Record the rank at the beginning of the iteration
        ranks->states[neuron_index] = neuron_model_get_rank_as_real(neuron);

        // Determine if a spike should occur
        bool spike = !has_sent_packets[neuron_index];

        if (spike) {
            // Tell the neuron model
            neuron_model_will_send_pkt(neuron);

            has_sent_packets[neuron_index] = true;

            // Get new rank
            payload_t broadcast_rank = neuron_model_get_broadcast_rank(neuron);

            // Do any required synapse processing
            synapse_dynamics_process_post_synaptic_event(time, neuron_index);

            // Record the spike
            out_spikes_set_spike(neuron_index);

            if (use_key) {

                // Wait until the expected time to send
                while (tc[T1_COUNT] > expected_time) {
                    // Do Nothing
                }
                expected_time -= time_between_spikes;

                // Send the spike
                key_t k = key | neuron_index;
                log_info("%16s[t=%04u|#%03d] Sending pkt  0x%08x=%k", "", time, neuron_index, k,
                    K(broadcast_rank));
                while (!spin1_send_mc_packet(k, broadcast_rank, WITH_PAYLOAD)) {
                    spin1_delay_us(1);
                }
            }
        } else {
            log_info("The neuron %d has been determined to not spike", neuron_index);
        }

        if (!has_sent_packets[neuron_index]) {
            hasSentAllPackets = false;
        }
    }

    // Disable interrupts to avoid possible concurrent access
    uint cpsr = spin1_int_disable();

    // Reset if all to true
    if (hasSentAllPackets) {
        bool hasReceivedAllUpdates = true;
        for (index_t neuron_index = 0; neuron_index < n_neurons; neuron_index++) {
            neuron_pointer_t neuron = &neuron_array[neuron_index];
            if ( !neuron_model_has_finished_iteration(neuron) ) {
                hasReceivedAllUpdates = false;
                break;
            }
        }

        if (hasReceivedAllUpdates) {
            _reset_has_sent_packets();
            for (index_t neuron_index = 0; neuron_index < n_neurons; neuron_index++) {
                neuron_pointer_t neuron = &neuron_array[neuron_index];
                neuron_model_iteration_did_finish(neuron);
            }
            _print_neurons();
        }
    }

    // record neuron state (membrane potential) if needed
    if (recording_is_channel_enabled(recording_flags, RANK_RECORDING_CHANNEL)) {
        n_recordings_outstanding += 1;
        ranks->time = time;
        recording_record_and_notify(
            RANK_RECORDING_CHANNEL, ranks, ranks_size, recording_done_callback);
    }

    // do logging stuff if required
    out_spikes_print();

    // Record any spikes this timestep
    if (recording_is_channel_enabled(recording_flags, SPIKE_RECORDING_CHANNEL)) {
        if (!out_spikes_is_empty()) {
            n_recordings_outstanding += 1;
            out_spikes_record(SPIKE_RECORDING_CHANNEL, time, recording_done_callback);
        }
    }

    // Re-enable interrupts
    spin1_mode_restore(cpsr);
}

void update_neuron_payload(uint32_t neuron_index, spike_t payload) {
    neuron_pointer_t neuron = &neuron_array[neuron_index];
    neuron_model_receive_packet(neuron_index, payload, neuron);
}
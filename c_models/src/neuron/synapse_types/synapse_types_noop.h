/*! \file
 * \brief implementation of synapse_types.h for Exponential shaping
*
* \details This is used to give a simple exponential decay to synapses.
*
* If we have combined excitatory/inhibitory synapses it will be
* because both excitatory and inhibitory synaptic time-constants
* (and thus propogators) are identical.
*/


#ifndef _SYNAPSE_TYPES_MY_IMPL_H_
#define _SYNAPSE_TYPES_MY_IMPL_H_

#include <debug.h>

// Number of bits required by the synapse type in the synapse row data structure
// Note: must match the number returned by the python method get_n_synapse_type_bits
#define SYNAPSE_TYPE_BITS 0

// Number of synapse types
// Note: must match the number returned by the python method get_n_synapse_types
#define SYNAPSE_TYPE_COUNT 0

// Parameters required to compute the synapse shape
// Note: # parameters must match # per neuron written by the python method write_synapse_parameters
typedef struct synapse_param_t {
} synapse_param_t;

// Include this here after defining the above items
#include <neuron/synapse_types/synapse_types.h>

//! \brief Shapes the values input into the neurons
//! \param[in] pointer to parameters the synapse parameter pointer passed in
//! \return Nothing
static inline void synapse_types_shape_input(
        synapse_param_pointer_t  __attribute__((__unused__)) parameter) {
}


//! \brief Adds the initial value to an input buffer for this shaping.  Allows
//         the input to be scaled before being added.
//! \param[in-out] input_buffers the pointer to the input buffers
//! \param[in] synapse_type_index the index of the synapse type to add the value to
//! \param[in] pointer to parameters the synapse parameters passed in
//! \param[in] input the input to be added
//! \return None
static inline void synapse_types_add_neuron_input(
        index_t  __attribute__((__unused__)) synapse_type_index,
        synapse_param_pointer_t  __attribute__((__unused__)) parameter,
        input_t  __attribute__((__unused__)) input) {
}

//! \brief Gets the excitatory input for a given neuron
//! \param[in] pointer to parameters the synapse parameters passed in
//! \return the excitatory input value
static inline input_t synapse_types_get_excitatory_input(
        synapse_param_pointer_t  __attribute__((__unused__)) parameter) {
    return 1;
}

//! \brief Gets the inhibitory input for a given neuron
//! \param[in] pointer to parameters the synapse parameters passed in
//! \return the inhibitory input value
static inline input_t synapse_types_get_inhibitory_input(
        synapse_param_pointer_t  __attribute__((__unused__)) parameter) {
    return 1;
}

//! \brief returns a human readable character for the type of synapse, for debug purposes
//! examples would be X = excitatory types, I = inhibitory types etc etc.
//! \param[in] synapse_type_index the synapse type index
//! \return a human readable character representing the synapse type.
static inline const char *synapse_types_get_type_char(
        index_t  __attribute__((__unused__)) synapse_type_index) {

    log_debug("Did not recognise synapse type %i", synapse_type_index);
    return "?";
}

//! \brief prints the input for a neuron id for debug purposes
//! \param[in] pointer to parameters the synapse parameters passed in
//! \return Nothing
static inline void synapse_types_print_input(
        synapse_param_pointer_t  __attribute__((__unused__)) parameter) {
}

//! \brief print parameters call
//! \param[in] parameter: the pointer to the parameters to print
//! \return Nothing
static inline void synapse_types_print_parameters(
        synapse_param_pointer_t  __attribute__((__unused__)) parameter) {
}

#endif  // _SYNAPSE_TYPES_MY_IMPL_H_

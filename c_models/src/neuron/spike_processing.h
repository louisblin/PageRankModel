#ifndef _SPIKE_PROCESSING_H_
#define _SPIKE_PROCESSING_H_

#include <common/neuron-typedefs.h>

bool spike_processing_initialise(
    size_t row_max_n_bytes, uint mc_pkt_callback_priority,
    uint user_event_priority, uint incoming_spike_buffer_size);

void spike_processing_finish_write(uint32_t process_id);

//! \brief returns the number of times the input buffer has overflowed
//! \return the number of times the input buffer has overflowed
uint32_t spike_processing_get_buffer_overflows();

payload_t spike_processing_payload_format(payload_t payload);
uint32_t spike_processing_increment_iteration_number(void);

#endif // _SPIKE_PROCESSING_H_

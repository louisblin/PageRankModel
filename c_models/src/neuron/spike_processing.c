#include "spike_processing.h"
#include "synapses.h"
#include <neuron/population_table/population_table.h>
#include <neuron/synapse_row.h>
#include <common/in_spikes.h>
#include <simulation.h>
#include <spin1_api.h>
#include <debug.h>

// The number of DMA Buffers to use
#define N_DMA_BUFFERS 2

// DMA tags
#define DMA_TAG_READ_SYNAPTIC_ROW 0
#define DMA_TAG_WRITE_PLASTIC_REGION 1

// DMA buffer structure combines the row read from SDRAM with
typedef struct dma_buffer {

    // Address in SDRAM to write back plastic region to
    address_t sdram_writeback_address;

    // Key of originating spike
    // (used to allow row data to be re-used for multiple spikes)
    spike_t originating_spike_key;
    spike_t originating_spike_payload;

    uint32_t n_bytes_transferred;

    // Row data
    uint32_t *row;

} dma_buffer;

extern uint32_t time;

// True if the DMA "loop" is currently running
static bool dma_busy;

// The DTCM buffers for the synapse rows
static dma_buffer dma_buffers[N_DMA_BUFFERS];

// The index of the next buffer to be filled by a DMA
static uint32_t next_buffer_to_fill;

// The index of the buffer currently being filled by a DMA read
static uint32_t buffer_being_read;

static uint32_t max_n_words;

static spike_t spike_pkt_key;
static spike_t spike_pkt_payload;

static uint32_t single_fixed_synapse[4];

/* PRIVATE FUNCTIONS - static for inlining */

static inline bool _add_key_payload(uint key, uint payload) {
    if(!in_spikes_add_spike(key)) {
        return false;
    }

    return in_spikes_add_spike(payload);
}

static inline bool _get_key_payload() {
    if (!in_spikes_get_next_spike(&spike_pkt_key)) {
        return false;
    }

    return in_spikes_get_next_spike(&spike_pkt_payload);
}

static inline void _do_dma_read(address_t row_address, size_t n_bytes_to_transfer) {
    log_debug("_do_dma_read: row_address[0]=%u | n_bytes_to_transfer=%u",
        ((uint32_t) row_address[0]), n_bytes_to_transfer);

    // Write the SDRAM address of the plastic region and the
    // Key of the originating spike to the beginning of DMA buffer
    dma_buffer *next_buffer = &dma_buffers[next_buffer_to_fill];
    next_buffer->sdram_writeback_address = row_address;
    next_buffer->originating_spike_key = spike_pkt_key;
    next_buffer->originating_spike_payload = spike_pkt_payload;
    next_buffer->n_bytes_transferred = n_bytes_to_transfer;

    // Start a DMA transfer to fetch this synaptic row into current buffer
    buffer_being_read = next_buffer_to_fill;
    spin1_dma_transfer(
        DMA_TAG_READ_SYNAPTIC_ROW, row_address, next_buffer->row, DMA_READ, n_bytes_to_transfer);
    next_buffer_to_fill = (next_buffer_to_fill + 1) % N_DMA_BUFFERS;
}


static inline void _do_direct_row(address_t row_address) {
    log_debug("_do_direct_row: row_address[0]=%u", ((uint32_t) row_address[0]));
    single_fixed_synapse[3] = (uint32_t) row_address[0];
    synapses_process_synaptic_row(time, single_fixed_synapse, false, 0);
}

static inline void _setup_synaptic_dma_read() {

    // Set up to store the DMA location and size to read
    address_t row_address;
    size_t n_bytes_to_transfer;

    bool setup_done = false;
    bool finished = false;
    uint cpsr = 0;
    while (!setup_done && !finished) {

        // If there's more rows to process from the previous spike
        while (!setup_done && population_table_get_next_address(
                &row_address, &n_bytes_to_transfer)) {

            // This is a direct row to process
            if (n_bytes_to_transfer == 0) {
                _do_direct_row(row_address);
            } else {
                _do_dma_read(row_address, n_bytes_to_transfer);
                setup_done = true;
            }
        }

        // If there's more incoming spikes
        cpsr = spin1_int_disable();
        while (!setup_done && _get_key_payload()) {
            spin1_mode_restore(cpsr);
            log_debug("Checking for row for spike %08x=%3.3k", spike_pkt_key,
                (REAL) spike_pkt_payload);

            // Decode spike to get address of destination synaptic row
            if (population_table_get_first_address(
                    spike_pkt_key, &row_address, &n_bytes_to_transfer)) {

                // This is a direct row to process
                if (n_bytes_to_transfer == 0) {
                    _do_direct_row(row_address);
                } else {
                    _do_dma_read(row_address, n_bytes_to_transfer);
                    setup_done = true;
                }
            }
            cpsr = spin1_int_disable();
        }

        if (!setup_done) {
            finished = true;
        }
        cpsr = spin1_int_disable();
    }

    // If the setup was not done, and there are no more spikes,
    // stop trying to set up synaptic DMAs
    if (!setup_done) {
        log_debug("DMA not busy");
        dma_busy = false;
    }
    spin1_mode_restore(cpsr);
}

static inline void _setup_synaptic_dma_write(uint32_t dma_buffer_index) {

    // Get pointer to current buffer
    dma_buffer *buffer = &dma_buffers[dma_buffer_index];

    // Get the number of plastic bytes and the write back address from the synaptic row
    size_t n_plastic_region_bytes = synapse_row_plastic_size(buffer->row) * sizeof(uint32_t);

    log_debug("Writing back %u bytes of plastic region to %08x",
              n_plastic_region_bytes, buffer->sdram_writeback_address + 1);

    // Start transfer
    spin1_dma_transfer(
        DMA_TAG_WRITE_PLASTIC_REGION, buffer->sdram_writeback_address + 1,
        synapse_row_plastic_region(buffer->row), DMA_WRITE, n_plastic_region_bytes);
}


/* CALLBACK FUNCTIONS - cannot be static */

// Called when a multicast packet is received
// pre-condition: packet
void _mcpl_packet_received_callback(uint key, uint payload) {
    log_info("[t=%04u|#%03d] Received pkt %08x=%3.3k", time, (0xff & (int) key), key,
        (REAL) payload);

    // If there was space to add spike to incoming spike queue
    // Note: assuming second add cannot fail as buffer size is a multiple of 2 x sizeof(uint32_t)
    if (_add_key_payload(key, payload)) {

        // If we're not already processing synaptic DMAs,
        // flag pipeline as busy and trigger a feed event
        if (!dma_busy) {

            log_debug("Sending user event for new spike");
            if (spin1_trigger_user_event(0, 0)) {
                dma_busy = true;
            } else {
                log_debug("Could not trigger user event\n");
            }
        }
    } else {
        log_debug("Could not add spike");
    }
}

// Called when a multicast packet is received
void _mc_packet_received_callback(uint key, uint payload) {
    use(payload);

    log_info("[IGNORED] Received spike %x at %d, DMA Busy = %d", key, time, dma_busy);

//    // If there was space to add spike to incoming spike queue
//    if (in_spikes_add_spike(key)) {
//
//        // If we're not already processing synaptic DMAs,
//        // flag pipeline as busy and trigger a feed event
//        if (!dma_busy) {
//
//            log_debug("Sending user event for new spike");
//            if (spin1_trigger_user_event(0, 0)) {
//                dma_busy = true;
//            } else {
//                log_debug("Could not trigger user event\n");
//            }
//        }
//    } else {
//        log_debug("Could not add spike");
//    }
}

// Called when a user event is received
void _user_event_callback(uint unused0, uint unused1) {
    use(unused0);
    use(unused1);
    _setup_synaptic_dma_read();
}

// Called when a DMA completes
void _dma_complete_callback(uint unused, uint tag) {
    use(unused);

    log_debug("DMA transfer complete with tag %u", tag);

    // Get pointer to current buffer
    uint32_t current_buffer_index = buffer_being_read;
    dma_buffer *current_buffer = &dma_buffers[current_buffer_index];

    // Start the next DMA transfer, so it is complete when we are finished
    _setup_synaptic_dma_read();

    // Process synaptic row repeatedly
    bool subsequent_spikes;
    do {

        // Are there any more incoming spikes from the same pre-synaptic neuron?
        subsequent_spikes = in_spikes_is_next_spike_equal(current_buffer->originating_spike_key);
        REAL payload = (REAL) current_buffer->originating_spike_payload;

        log_debug("synapses_process_synaptic_row(%d, 0x%08x, %3.3k, %x, %x)", time,
            current_buffer->row, payload, !subsequent_spikes, current_buffer_index);

        // Process synaptic row, writing it back if it's the last time it's going to be processed
        if (!synapses_process_synaptic_row_page_rank(
                time, current_buffer->row, payload, !subsequent_spikes, current_buffer_index)) {
            log_error("Error processing spike 0x%.8x=%3.3k for address 0x%.8x (local=0x%.8x)",
                current_buffer->originating_spike_key, payload,
                current_buffer->sdram_writeback_address, current_buffer->row);

            // Print out the row for debugging
            for (uint32_t i = 0; i < (current_buffer->n_bytes_transferred >> 2); i++) {
                log_error("%u: 0x%.8x", i, current_buffer->row[i]);
            }

            rt_error(RTE_SWERR);
        }
    } while (subsequent_spikes);
}


/* INTERFACE FUNCTIONS - cannot be static */

bool spike_processing_initialise(
        size_t row_max_n_words, uint mc_packet_callback_priority,
        uint user_event_priority, uint incoming_spike_buffer_size) {

    // Check priority is -1, i.e. callback cannot be preempted
    if (mc_packet_callback_priority != ((uint) -1)) {
        log_error("mc_packet_callback_priority = %u != -1: callback could be preempted",
            mc_packet_callback_priority);
    }

    // Allocate the DMA buffers
    for (uint32_t i = 0; i < N_DMA_BUFFERS; i++) {
        dma_buffers[i].row = (uint32_t*) spin1_malloc(row_max_n_words * sizeof(uint32_t));
        if (dma_buffers[i].row == NULL) {
            log_error("Could not initialise DMA buffers");
            return false;
        }
        log_info("DMA buffer %u allocated at 0x%08x", i, dma_buffers[i].row);
    }
    dma_busy = false;
    next_buffer_to_fill = 0;
    buffer_being_read = N_DMA_BUFFERS;
    max_n_words = row_max_n_words;

    // Allocate incoming spike buffer
    if (!in_spikes_initialize_spike_buffer(incoming_spike_buffer_size)) {
        return false;
    }

    // Set up for single fixed synapses (data that is consistent per direct row)
    single_fixed_synapse[0] = 0;
    single_fixed_synapse[1] = 1;
    single_fixed_synapse[2] = 0;

    // Set up the callbacks
    spin1_callback_on(MCPL_PACKET_RECEIVED,
            _mcpl_packet_received_callback, mc_packet_callback_priority);
    spin1_callback_on(MC_PACKET_RECEIVED,
            _mc_packet_received_callback, mc_packet_callback_priority);
    simulation_dma_transfer_done_callback_on(DMA_TAG_READ_SYNAPTIC_ROW, _dma_complete_callback);
    spin1_callback_on(USER_EVENT, _user_event_callback, user_event_priority);

    return true;
}

void spike_processing_finish_write(uint32_t process_id) {
    _setup_synaptic_dma_write(process_id);
}

//! \brief returns the number of times the input buffer has overflowed
//! \return the number of times the input buffer has overloaded
uint32_t spike_processing_get_buffer_overflows() {

    // Check for buffer overflow
    return in_spikes_get_n_buffer_overflows();
}

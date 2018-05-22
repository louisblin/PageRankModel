#ifndef _MY_THRESHOLD_TYPE_H_
#define _MY_THRESHOLD_TYPE_H_

#include <neuron/threshold_types/threshold_type.h>

typedef struct threshold_type_t {
    uint32_t incoming_edges_count;

} threshold_type_t;

static inline bool threshold_type_is_above_threshold(state_t value,
        threshold_type_pointer_t threshold_type) {

    log_debug("real_compare(%3.3k, >=, %03d)", value, threshold_type->incoming_edges_count);

    // Return true or false depending on if the threshold has been reached
    return REAL_COMPARE(value, >=, threshold_type->incoming_edges_count);
}

#endif // _MY_THRESHOLD_TYPE_H_

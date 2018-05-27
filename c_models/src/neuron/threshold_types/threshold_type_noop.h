#ifndef _MY_THRESHOLD_TYPE_H_
#define _MY_THRESHOLD_TYPE_H_

#include <neuron/threshold_types/threshold_type.h>

typedef struct threshold_type_t {
} threshold_type_t;

static inline bool threshold_type_is_above_threshold(state_t value,
        threshold_type_pointer_t threshold_type) {
    use(threshold_type);
    use(value);
    return true;
}

#endif // _MY_THRESHOLD_TYPE_H_

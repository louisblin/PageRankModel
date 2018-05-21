from spynnaker.pyNN.utilities import utility_calls
from spynnaker.pyNN.models.neural_properties import NeuronParameter
from data_specification.enums import DataType
from spynnaker.pyNN.models.neuron.threshold_types import AbstractThresholdType

from enum import Enum


class _THRESHOLD_TYPES(Enum):

    INCOMING_EDGES_COUNT = (1, DataType.UINT32)

    def __new__(cls, value, data_type):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._data_type = data_type
        return obj

    @property
    def data_type(self):
        return self._data_type


class ThresholdTypePageRank(AbstractThresholdType):
    """ A threshold that is a static value
    """
    def __init__(self, n_neurons, incoming_edges_count):
        AbstractThresholdType.__init__(self)
        self._n_neurons = n_neurons

        # Parameters
        self._incoming_edges_count = utility_calls.convert_param_to_numpy(
            incoming_edges_count, n_neurons)

    # Getters and setters for the parameters

    @property
    def incoming_edges_count(self):
        return self._incoming_edges_count

    @incoming_edges_count.setter
    def incoming_edges_count(self, incoming_edges_count):
        self._incoming_edges_count = utility_calls.convert_param_to_numpy(
            incoming_edges_count, self._n_neurons)

    # Mapping per-neuron parameters (`threshold_type_t' in C code)

    def get_n_threshold_parameters(self):
        return len(_THRESHOLD_TYPES)

    def get_threshold_parameters(self):
        # Note: must match the order of the parameters in the `threshold_type_t' in the C code
        return [
            NeuronParameter(getattr(self, '_'+item.name.lower()), item.data_type)
            for item in _THRESHOLD_TYPES
        ]

    def get_threshold_parameter_types(self):
        return [item.data_type for item in _THRESHOLD_TYPES]

    def get_n_cpu_cycles_per_neuron(self):
        # TODO: update to the number of cycles used by threshold_type_is_above_threshold
        #   Note: This can be guessed
        return 10

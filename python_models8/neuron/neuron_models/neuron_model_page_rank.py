import numpy as np
from decimal import Decimal
from enum import Enum

from pacman.executor.injection_decorator import inject_items
from pacman.model.decorators.overrides import overrides
from spynnaker.pyNN.models.neural_properties import NeuronParameter
from spynnaker.pyNN.models.abstract_models import AbstractContainsUnits
from spynnaker.pyNN.models.neuron.neuron_models import AbstractNeuronModel
from spynnaker.pyNN.utilities import utility_calls
from data_specification.enums import DataType


class _NEURAL_PARAMETERS(Enum):
    INCOMING_EDGES_COUNT = (1, DataType.UINT32, 'count')
    OUTGOING_EDGES_COUNT = (2, DataType.UINT32, 'count')
    RANK_INIT = (3, DataType.U032, 'rk')
    CURR_RANK_ACC_INIT = (4, DataType.U032, 'rk')
    CURR_RANK_COUNT_INIT = (5, DataType.UINT32, 'count')
    ITER_STATE_INIT = (6, DataType.UINT32, 'state')

    def __new__(cls, value, data_type, unit):
        obj = object.__new__(cls)
        obj._value_ = value  # Note: value order is used for iteration
        obj._data_type = data_type
        obj._unit = unit
        return obj

    @property
    def data_type(self):
        return self._data_type

    @property
    def unit(self):
        return self._unit


def convert_rank(n):
    # Rank (voltage originally was stored as a DataType.S1615)
    scale = Decimal(_NEURAL_PARAMETERS.RANK_INIT.data_type.scale / DataType.S1615.scale)
    n = np.float(Decimal(n) / scale)
    # Handle erroneous conversion of non-existing sign bit in UFRACT
    if n < 0:
        n = 1 + n
    return n


class NeuronModelPageRank(AbstractNeuronModel, AbstractContainsUnits):

    def __init__(self, n_neurons,
                 incoming_edges_count, outgoing_edges_count,
                 rank_init, curr_rank_acc_init, curr_rank_count_init, iter_state_init):
        AbstractNeuronModel.__init__(self)
        AbstractContainsUnits.__init__(self)

        self._n_neurons = n_neurons

        # Store any parameters
        self._incoming_edges_count = self._var_init(incoming_edges_count)
        self._outgoing_edges_count = self._var_init(outgoing_edges_count)

        # Store any state variables
        self._initialize_state_vars([
            ('rank_init', rank_init),
            ('curr_rank_acc_init', curr_rank_acc_init),
            ('curr_rank_count_init', curr_rank_count_init),
            ('iter_state_init', iter_state_init),
        ])

    def _var_init(self, state_var):
        return utility_calls.convert_param_to_numpy(state_var, self._n_neurons)

    # Getters and setters for the parameters
    @property
    def incoming_edges_count(self):
        return self._incoming_edges_count

    @incoming_edges_count.setter
    def incoming_edges_count(self, incoming_edges_count):
        self._incoming_edges_count = self._var_init(incoming_edges_count)

    @property
    def outgoing_edges_count(self):
        return self._outgoing_edges_count

    @outgoing_edges_count.setter
    def outgoing_edges_count(self, outgoing_edges_count):
        self._outgoing_edges_count = self._var_init(outgoing_edges_count)

    # Initializers for the state variables
    def _initialize_state_vars(self, state_vars):
        def _mk_initialize(state_var):
            def initialize(self, val):
                self[state_var] = self._var_init(val)
            return initialize

        for name, val in state_vars:
            _state_var = '_{}'.format(name)
            initialize_name = 'initialize_{}'.format(name[:-5])

            setattr(self, _state_var, self._var_init(val))
            setattr(self, initialize_name, _mk_initialize(_state_var))

    # Mapping per-neuron parameters (`neuron_t' in C code)

    @overrides(AbstractNeuronModel.get_n_neural_parameters)
    def get_n_neural_parameters(self):
        return len(_NEURAL_PARAMETERS)

    @overrides(AbstractNeuronModel.get_neural_parameters)
    def get_neural_parameters(self):
        # Note: must match the order of the parameters in the `neuron_t' in the C code
        return [
            NeuronParameter(getattr(self, '_'+item.name.lower()), item.data_type)
            for item in _NEURAL_PARAMETERS
        ]

    @overrides(AbstractNeuronModel.get_neural_parameter_types)
    def get_neural_parameter_types(self):
        return [item.data_type for item in _NEURAL_PARAMETERS]

    # Mapping population-wide parameters (`global_neuron_t' in C code)

    @overrides(AbstractNeuronModel.get_n_global_parameters)
    def get_n_global_parameters(self):
        return 1

    # noinspection PyMethodOverriding
    @inject_items({"machine_time_step": "MachineTimeStep"})
    @overrides(
        AbstractNeuronModel.get_global_parameters,
        additional_arguments={"machine_time_step"}
    )
    def get_global_parameters(self, machine_time_step):
        # Note: must match the order of the parameters in the `global_neuron_t' in the C code
        return [
            # uint32_t machine_time_step
            NeuronParameter(machine_time_step, DataType.UINT32)
        ]

    @overrides(AbstractNeuronModel.get_global_parameter_types)
    def get_global_parameter_types(self):
        return [DataType.UINT32]

    @overrides(AbstractNeuronModel.get_n_cpu_cycles_per_neuron)
    def get_n_cpu_cycles_per_neuron(self):
        # Number of CPU cycles taken by neuron_model functions in main loop
        #   Note: This can be a guess
        return 40

    @overrides(AbstractContainsUnits.get_units)
    def get_units(self, variable):
        return _NEURAL_PARAMETERS[variable.upper()].unit

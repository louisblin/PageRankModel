from pacman.executor.injection_decorator import inject_items
from pacman.model.decorators.overrides import overrides
from spynnaker.pyNN.models.neural_properties import NeuronParameter
from spynnaker.pyNN.models.abstract_models import AbstractContainsUnits
from spynnaker.pyNN.models.neuron.neuron_models import AbstractNeuronModel
from spynnaker.pyNN.utilities import utility_calls
from data_specification.enums import DataType

from enum import Enum


class _NEURAL_PARAMETERS(Enum):
    RANK_INIT = (1, DataType.S1615, 'rk')
    CURR_RANK_ACC_INIT = (2, DataType.S1615, 'rk')
    CURR_RANK_COUNT_INIT = (3, DataType.S1615, 'au')

    def __new__(cls, value, data_type, unit):
        obj = object.__new__(cls)
        # Note: value order is used for iteration
        obj._value_ = value
        obj._data_type = data_type
        obj._unit = unit
        return obj

    @property
    def data_type(self):
        return self._data_type

    @property
    def unit(self):
        return self._unit


class NeuronModelPageRank(AbstractNeuronModel, AbstractContainsUnits):

    def __init__(self, n_neurons, rank_init, curr_rank_acc_init, curr_rank_count_init):
        AbstractNeuronModel.__init__(self)
        AbstractContainsUnits.__init__(self)

        self._n_neurons = n_neurons

        # Store any parameters

        # Store any state variables
        self._initialize_state_vars([
            ('rank_init', rank_init),
            ('curr_rank_acc_init', curr_rank_acc_init),
            ('curr_rank_count_init', curr_rank_count_init),
        ])

    # Getters and setters for the parameters

    # Initializers for the state variables
    def _initialize_state_vars(self, state_vars):
        def _state_init(state_var):
            return utility_calls.convert_param_to_numpy(state_var, self._n_neurons)

        def _mk_initialize(state_var):
            def initialize(self, val):
                self[state_var] = _state_init(val)
            return initialize

        for name, val in state_vars:
            _state_var = '_{}'.format(name)
            initialize_name = 'initialize_{}'.format(name[:-5])

            setattr(self, _state_var, _state_init(val))
            setattr(self, initialize_name, _mk_initialize(_state_var))

    # Required for obj[...] access of `setattr' methods
    def __getitem__(self, key):
        return getattr(self, key)

    # Mapping per-neuron parameters (`neuron_t' in C code)
    @overrides(AbstractNeuronModel.get_n_neural_parameters)
    def get_n_neural_parameters(self):
        return len(_NEURAL_PARAMETERS)

    @overrides(AbstractNeuronModel.get_neural_parameters)
    def get_neural_parameters(self):
        # Note: must match the order of the parameters in the `neuron_t' in the C code
        return [
            NeuronParameter(self['_{}'.format(item.name.lower())], item.data_type)
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
        # TODO: update with the number of CPU cycles taken by the neuron_model_state_update,
        #   neuron_model_get_membrane_voltage and neuron_model_has_spiked functions in the C code
        #   Note: This can be a guess
        return 80

    @overrides(AbstractContainsUnits.get_units)
    def get_units(self, variable):
        return _NEURAL_PARAMETERS[variable.upper()].unit

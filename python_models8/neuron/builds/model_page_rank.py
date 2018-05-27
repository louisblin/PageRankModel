# main interface to use the spynnaker related tools.
# ALL MODELS MUST INHERIT FROM THIS
from spynnaker.pyNN.models.neuron import AbstractPopulationVertex
from spynnaker.pyNN.models.neuron.input_types import InputTypeCurrent
from python_models8.neuron.neuron_models.neuron_model_page_rank import NeuronModelPageRank
from python_models8.neuron.synapse_types.synapse_type_noop import SynapseTypeNoOp
from python_models8.neuron.threshold_types.threshold_type_noop import ThresholdTypeNoOp


class PageRankBase(AbstractPopulationVertex):

    # TODO: Set the maximum number of atoms per core that can be supported.
    # For more complex models, you might need to reduce this number.
    _model_based_max_atoms_per_core = 256

    # Default parameters for this build, used when end user has not entered any
    default_parameters = {
        'incoming_edges_count': 0,
        'outgoing_edges_count': 0,
    }

    none_pynn_default_parameters = {
        'rank_init': 1,
        'curr_rank_acc_init': 0,
        'curr_rank_count_init': 0,
        'has_completed_iter_init': 0,
    }

    def __init__(
            self, n_neurons, spikes_per_second=AbstractPopulationVertex.
            none_pynn_default_parameters['spikes_per_second'],
            ring_buffer_sigma=AbstractPopulationVertex.
            none_pynn_default_parameters['ring_buffer_sigma'],
            incoming_spike_buffer_size=AbstractPopulationVertex.
            none_pynn_default_parameters['incoming_spike_buffer_size'],
            constraints=AbstractPopulationVertex.none_pynn_default_parameters['constraints'],
            label=AbstractPopulationVertex.none_pynn_default_parameters['label'],

            # Model parameters
            incoming_edges_count=default_parameters['incoming_edges_count'],
            outgoing_edges_count=default_parameters['outgoing_edges_count'],

            # Threshold types parameters

            # Initial values for the state variables; this is not technically done in PyNN
            rank_init=none_pynn_default_parameters['rank_init'],
            curr_rank_acc_init=none_pynn_default_parameters['curr_rank_acc_init'],
            curr_rank_count_init=none_pynn_default_parameters['curr_rank_count_init'],
            has_completed_iter_init=none_pynn_default_parameters['has_completed_iter_init']):

        neuron_model = NeuronModelPageRank(
                n_neurons,
                incoming_edges_count, outgoing_edges_count,
                rank_init, curr_rank_acc_init, curr_rank_count_init, has_completed_iter_init)

        input_type = InputTypeCurrent()

        synapse_type = SynapseTypeNoOp()

        threshold_type = ThresholdTypeNoOp()

        # instantiate the sPyNNaker system by initialising the AbstractPopulationVertex
        AbstractPopulationVertex.__init__(
            # standard inputs, do not need to change.
            self, n_neurons=n_neurons, label=label,
            spikes_per_second=spikes_per_second,
            ring_buffer_sigma=ring_buffer_sigma,
            incoming_spike_buffer_size=incoming_spike_buffer_size,
            max_atoms_per_core=PageRankBase._model_based_max_atoms_per_core,

            # These are the various model types
            neuron_model=neuron_model, input_type=input_type,
            synapse_type=synapse_type, threshold_type=threshold_type,
            additional_input=None,
            model_name="PageRank", # name shown in reports
            binary="page_rank.aplx") # c src binary name

    @staticmethod
    def get_max_atoms_per_core():
        return PageRankBase._model_based_max_atoms_per_core

    @staticmethod
    def set_max_atoms_per_core(new_value):
        PageRankBase._model_based_max_atoms_per_core = new_value

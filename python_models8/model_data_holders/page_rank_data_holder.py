# main interface to use the spynnaker related tools.
# ALL MODELS MUST INHERIT FROM THIS
from spynnaker.pyNN.models.neuron import AbstractPopulationVertex
from spynnaker8.utilities import DataHolder
from python_models8.neuron.builds.model_page_rank import PageRankBase


class PageRankDataHolder(DataHolder):
    def __init__(
            self,

            # AbstractPopulationVertex
            spikes_per_second=(
                    AbstractPopulationVertex.none_pynn_default_parameters['spikes_per_second']),
            ring_buffer_sigma=(
                    AbstractPopulationVertex.none_pynn_default_parameters['ring_buffer_sigma']),
            incoming_spike_buffer_size=(
                    AbstractPopulationVertex.none_pynn_default_parameters[
                        'incoming_spike_buffer_size']),
            constraints=AbstractPopulationVertex.none_pynn_default_parameters['constraints'],
            label=AbstractPopulationVertex.none_pynn_default_parameters['label'],

            # PageRankBase
            rank_init=PageRankBase.none_pynn_default_parameters['rank_init'],
            curr_rank_acc_init=PageRankBase.none_pynn_default_parameters['curr_rank_acc_init'],
            curr_rank_count_init=PageRankBase.none_pynn_default_parameters['curr_rank_count_init'],
            v_thresh=PageRankBase.default_parameters['v_thresh'],
            tau_syn_E=PageRankBase.default_parameters['tau_syn_E'],
            tau_syn_I=PageRankBase.default_parameters['tau_syn_I'],
            isyn_exc=PageRankBase.default_parameters['isyn_exc'],
            isyn_inh=PageRankBase.default_parameters['isyn_inh']):
        DataHolder.__init__(
            self, {
                'spikes_per_second': spikes_per_second,
                'ring_buffer_sigma': ring_buffer_sigma,
                'incoming_spike_buffer_size': incoming_spike_buffer_size,
                'constraints': constraints,
                'label': label,
                'v_thresh': v_thresh,
                'tau_syn_E': tau_syn_E, 'tau_syn_I': tau_syn_I,
                'isyn_exc': isyn_exc, 'isyn_inh': isyn_inh,
                'rank_init': rank_init,
                'curr_rank_acc_init': curr_rank_acc_init,
                'curr_rank_count_init': curr_rank_count_init,
            }
        )

    @staticmethod
    def build_model():
        return PageRankBase

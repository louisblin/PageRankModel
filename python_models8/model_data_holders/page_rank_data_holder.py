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
            damping_factor=PageRankBase.default_parameters['damping_factor'],
            damping_sum=PageRankBase.default_parameters['damping_sum'],
            incoming_edges_count=PageRankBase.default_parameters['incoming_edges_count'],
            outgoing_edges_count=PageRankBase.default_parameters['outgoing_edges_count'],
            rank_init=PageRankBase.none_pynn_default_parameters['rank_init'],
            curr_rank_acc_init=PageRankBase.none_pynn_default_parameters['curr_rank_acc_init'],
            curr_rank_count_init=PageRankBase.none_pynn_default_parameters['curr_rank_count_init'],
            iter_state_init=PageRankBase.none_pynn_default_parameters['iter_state_init']):
        DataHolder.__init__(
            self, {
                'spikes_per_second': spikes_per_second,
                'ring_buffer_sigma': ring_buffer_sigma,
                'incoming_spike_buffer_size': incoming_spike_buffer_size,
                'constraints': constraints,
                'label': label,
                'damping_factor': damping_factor,
                'damping_sum': damping_sum,
                'incoming_edges_count': incoming_edges_count,
                'outgoing_edges_count': outgoing_edges_count,
                'rank_init': rank_init,
                'curr_rank_acc_init': curr_rank_acc_init,
                'curr_rank_count_init': curr_rank_count_init,
                'iter_state_init': iter_state_init,
            }
        )

    @staticmethod
    def build_model():
        return PageRankBase

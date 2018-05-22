import argparse
import sys

import spynnaker8 as p

from python_models8.model_data_holders.page_rank_data_holder import PageRankDataHolder as Page_Rank
from python_models8.synapse_dynamics.synapse_dynamics_noop import SynapseDynamicsNoOp
from examples.utils import draw_input_graph, draw_output_graph

RANK = 'v'


def run(show_in=False, show_out=False):
    ###############################################################################
    # Simulation parameters

    run_time = 5
    time_step = 1.0
    n_neurons = 3

    p.setup(time_step, min_delay=time_step)

    ###############################################################################
    # Construct simulation graph

    # Compute links
    injection_connections = []
    incoming_edges_count = [0] * n_neurons
    for src in range(n_neurons):
        tgt = (src + 1) % n_neurons
        injection_connections.append((src, tgt))
        incoming_edges_count[tgt] += 1

    # Vertices
    pop = p.Population(
        n_neurons,
        Page_Rank(incoming_edges_count=incoming_edges_count),
        label="page_rank"
    )

    # Edges
    p.Projection(
        pop, pop,
        p.FromListConnector(injection_connections),
        synapse_type=SynapseDynamicsNoOp()
    )

    models = [pop]

    ###############################################################################
    # Run simulation / report

    if show_in:
        draw_input_graph(injection_connections)

    for m in models:
        m.record([RANK])

    p.run(run_time)

    # Graph reporting
    if show_out:
        draw_output_graph(models, RANK, run_time)
    else:
        raw_input('Press any key to finish...')

    p.end()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample page rank graph with 4 vertices')
    parser.add_argument('--show-in', action='store_true', help='Display input directed graph.')
    parser.add_argument('--show-out', action='store_true', help='Display output ranks curves.')

    sys.exit(run(**vars(parser.parse_args())))
import argparse
import sys

import spynnaker8 as p

import examples.utils as utils

RANK = 'v'


def run(show_in=False, show_out=False):
    ###############################################################################
    # Simulation parameters

    n_neurons = 4
    run_time = 10.

    parameters = dict(
        n_neurons=n_neurons,
        run_time=run_time,
        timestep=1.,
        time_scale_factor=4
    )
    p.setup(**parameters)

    ###############################################################################
    # Construct simulation graph
    # From: https://www.youtube.com/watch?v=P8Kt6Abq_rM

    vertices = list(range(n_neurons))

    # Compute links
    [A, B, C, D] = vertices
    edges = [
        (A, B),
        (A, C),
        (B, D),
        (C, A),
        (C, B),
        (C, D),
        (D, C),
    ]

    model = utils.create_page_rank_model(p, vertices, edges)

    ###############################################################################
    # Run simulation / report

    if show_in:
        utils.draw_input_graph(edges)

    model.record([RANK])

    p.run(run_time)

    # Graph reporting
    utils.draw_output_graph(model, RANK, run_time, show_graph=show_out)
    # utils.python_page_rank(vertices, edges, show=True)

    if not show_out:
        raw_input('Press any key to finish...')

    p.end()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample page rank graph with 4 vertices')
    parser.add_argument('--show-in', action='store_true', help='Display input directed graph.')
    parser.add_argument('--show-out', action='store_true', help='Display output ranks curves.')

    sys.exit(run(**vars(parser.parse_args())))
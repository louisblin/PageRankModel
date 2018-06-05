import argparse
import networkx as nx
import random
import sys
import tqdm

from examples.page_rank import PageRankSimulation, LOG_LEVEL_PAGE_RANK_INFO

N_ITER = 15
timestep = .1
RUN_TIME = N_ITER * timestep
PARAMETERS = {
    # Experimental results of good values
    # |V|=400  |E|=600   : ts=1. tsf=40
    # |V|=800  |E|=1200  : ts=1. tsf=45
    # |V|=1600 |E|=2400  : ts=1. tsf=150
    # |V|=3200 |E|=4800  : ts=?  tsf=?
    'time_scale_factor': 20000,
}


def _mk_label(n):
    return '#%d' % n


def _mk_rd_node(node_count):
    return random.randint(0, node_count - 1)


def _mk_graph(node_count, edge_count):
    # Under these constraints we can comply with the requirements below
    assert node_count <= edge_count <= node_count**2, \
        "Need node_count=%d < edge_count=%d < %d " % (node_count, edge_count, node_count**2)

    # Ensures no double edges
    edges = set([])

    # Ensures no dangling nodes
    for i in range(node_count):
        edges.add((i, _mk_rd_node(node_count)))

    for _ in tqdm.tqdm(range(node_count, edge_count), desc="Generating edges",
                       initial=node_count, total=edge_count):
        while True:
            l = len(edges)
            edges.add((_mk_rd_node(node_count), _mk_rd_node(node_count)))
            if len(edges) > l:
                break  # Only move to next iteration if we've added a new edge

    edges = [(_mk_label(src), _mk_label(tgt))
             for src, tgt in tqdm.tqdm(edges, desc="Formatting edges")]

    return edges


def _mk_sim_run(node_count=None, edge_count=None, verify=False, pause=False, show_out=False):
    ###############################################################################
    # Create random Page Rank graphs
    labels = map(_mk_label, list(range(node_count)))
    edges = _mk_graph(node_count, edge_count)

    ###############################################################################
    # Run simulation / report
    with PageRankSimulation(RUN_TIME, edges, labels, PARAMETERS, log_level=0, pause=pause) as sim:
        is_correct = sim.run(verify=verify, diff_only=True)
        sim.draw_output_graph(show_graph=show_out)
        return is_correct


def run(runs=None, **kwargs):
    errors = 0
    for _ in tqdm.tqdm(range(runs), total=runs):
        while True:
            try:
                is_correct = _mk_sim_run(**kwargs)
                errors += 0 if is_correct else 1
                break
            except nx.PowerIterationFailedConvergence:
                print('Skipping nx.PowerIterationFailedConvergence graph...')

    print('Finished robustness test with %d/%d error(s).' % (errors, runs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create random Page Rank graphs')
    parser.add_argument('-r', '--runs', type=int, default=1, help='# runs. Default is 1.')
    parser.add_argument('node_count', metavar='NODE_COUNT', type=int, help='# nodes per graph')
    parser.add_argument('edge_count', metavar='EDGE_COUNT', type=int, help='# edges per graph')
    parser.add_argument('-v', '--verify', action='store_true', help='Verify sim w/ Python PR impl')
    parser.add_argument('-p', '--pause', action='store_true', help='Pause after each runs')
    parser.add_argument('-o', '--show-out', action='store_true', help='Display ranks curves output')

    random.seed(42)
    sys.exit(run(**vars(parser.parse_args())))

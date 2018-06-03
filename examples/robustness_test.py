import argparse
import networkx as nx
import random
import sys
import tqdm

from examples.page_rank import PageRankSimulation, LOG_LEVEL_PAGE_RANK_INFO

N_ITER = 20.
timestep = 100.
RUN_TIME = N_ITER * timestep
PARAMETERS = {
    'time_scale_factor': 10.,
    'timestep': timestep,
    'min_delay': timestep,
    'max_delay': timestep
}


def _mk_label(n):
    return '#%d' % n


def _mk_rd_node(node_count):
    return _mk_label(random.randint(0, node_count - 1))


def _mk_graph(node_count, edge_count):
    # Under these constraints we can comply with the requirements below
    assert node_count <= edge_count <= node_count**2, \
        "Need node_count=%d < edge_count=%d < %d " % (node_count, edge_count, node_count**2)

    edges = []

    # Ensures no dangling nodes
    for i in range(node_count):
        edges.append((_mk_label(i), _mk_rd_node(node_count)))

    for _ in range(node_count, edge_count):
        while True:
            edge = (_mk_rd_node(node_count), _mk_rd_node(node_count))
            # Ensures no double edges
            if edge not in edges:
                edges.append(edge)
                break
    return edges

def _mk_sim_run(node_count, edge_count, pause_incorrect, show_out):
    ###############################################################################
    # Create random Page Rank graphs
    labels = map(_mk_label, list(range(node_count)))
    edges = _mk_graph(node_count, edge_count)

    ###############################################################################
    # Run simulation / report
    with PageRankSimulation(RUN_TIME, edges, labels=labels, parameters=PARAMETERS,
                            log_level=10) as sim:
        is_correct = sim.run(verify=True)
        sim.draw_output_graph(show_graph=show_out, pause=(not is_correct) and pause_incorrect)
        return is_correct


def run(runs=None, node_count=None, edge_count=None, pause_incorrect=False, show_out=False):
    errors = 0
    for _ in tqdm.tqdm(range(runs), total=runs):
        while True:
            try:
                is_correct = _mk_sim_run(node_count, edge_count, pause_incorrect, show_out)
                errors += 0 if is_correct else 1
                break
            except nx.PowerIterationFailedConvergence:
                print('Skipping nx.PowerIterationFailedConvergence graph...')

    print('Finished robustness test with %d/%d error(s).' % (errors, runs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create random Page Rank graphs')
    parser.add_argument('-r', '--runs', type=int, default=10, help='# runs')
    parser.add_argument('node_count', metavar='NODE_COUNT', type=int, help='# nodes per graph')
    parser.add_argument('edge_count', metavar='EDGE_COUNT', type=int, help='# edges per graph')
    parser.add_argument('--pause-incorrect', action='store_true', help='Pause after incorrect runs')
    parser.add_argument('--show-out', action='store_true', help='Display ranks curves output')

    random.seed(42)
    sys.exit(run(**vars(parser.parse_args())))

import argparse
import networkx as nx
import random
import sys
import tqdm

from examples.page_rank import PageRankSimulation, silence_stdout

RUN_TIME = 91.
PARAMETERS = {
    'time_scale_factor': 4
}


def _mk_label(n):
    return '#%d' % n


def _mk_rd_node(node_count):
    return _mk_label(random.randint(0, node_count - 1))


def _mk_graph(node_count, edge_count):
    # Under these constraints we can comply with the requirements below
    assert node_count < edge_count < node_count**2, \
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


def _mk_sim_run(node_count, edge_count, show_incorrect):
    ###############################################################################
    # Create random Page Rank graphs
    labels = map(_mk_label, list(range(node_count)))
    edges = _mk_graph(node_count, edge_count)

    ###############################################################################
    # Run simulation / report
    with PageRankSimulation(RUN_TIME, edges, labels=labels, parameters=PARAMETERS) as sim:
        with silence_stdout():
            is_correct, msg = sim.run(verify=True, get_string=True, find_iter=True)
        if not is_correct:
            sim.draw_input_graph(show_graph=show_incorrect)
        return is_correct, msg


def run(runs=None, node_count=None, edge_count=None, show_incorrect=False):
    errors = 0
    for _ in tqdm.tqdm(range(runs), total=runs):
        while True:
            try:
                is_correct, msg = _mk_sim_run(node_count, edge_count, show_incorrect)
            except nx.PowerIterationFailedConvergence:
                print('Skipping nx.PowerIterationFailedConvergence graph...')
                continue
            errors += 0 if is_correct else 1
            print(msg)
            break

    print('Finished robustness test with %d/%d error(s).' % (errors, runs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create random Page Rank graphs')
    parser.add_argument('-r', '--runs', type=int, default=10, help='# runs')
    parser.add_argument('node_count', metavar='NODE_COUNT', type=int, help='# nodes per graph')
    parser.add_argument('edge_count', metavar='EDGE_COUNT', type=int, help='# edges per graph')
    parser.add_argument('--show-incorrect', action='store_true',
                        help='Display ranks curves output for incorrect runs.')

    random.seed(42)
    sys.exit(run(**vars(parser.parse_args())))

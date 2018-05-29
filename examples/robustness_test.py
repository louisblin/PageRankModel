import argparse
import random
import sys
import tqdm

from examples.page_rank import PageRankSimulation, silence_stdout

RUN_TIME = 10.
PARAMETERS = {
    'time_scale_factor': 10
}


def run(runs=None, node_count=None, edge_count=None, show_incorrect=False):
    assert node_count < edge_count, "Need node_count=%d < edge_count=%d" % (node_count, edge_count)

    mk_label = lambda n: '#%d' % n
    mk_rd_node = lambda: mk_label(random.randint(0, node_count - 1))
    errors = 0

    for _ in tqdm.tqdm(range(runs), total=runs):
        ###############################################################################
        # Create random Page Rank graphs
        labels = map(mk_label, list(range(node_count)))
        edges = []
        # Ensure no dangling nodes
        for i in range(node_count):
            edges.append((mk_label(i), mk_rd_node()))
        for _ in range(node_count, edge_count):
            while True:
                edge = (mk_rd_node(), mk_rd_node())
                if not (edge_count in edges):
                    edges.append(edge)
                    break

        ###############################################################################
        # Run simulation / report
        with PageRankSimulation(RUN_TIME, edges, labels=labels, parameters=PARAMETERS) as sim:
            with silence_stdout():
                is_correct, msg = sim.run(verify=True, get_string=True)
            print(msg)
            if not is_correct:
                sim.draw_input_graph(show_graph=show_incorrect)
                errors += 1

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

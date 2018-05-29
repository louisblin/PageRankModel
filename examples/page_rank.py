import os
import sys
from contextlib import contextmanager

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import spynnaker8 as p
from prettytable import PrettyTable
from pyNN.utility.plotting import Figure, Panel

from python_models8.model_data_holders.page_rank_data_holder import PageRankDataHolder as Page_Rank
from python_models8.synapse_dynamics.synapse_dynamics_noop import SynapseDynamicsNoOp

RANK = 'v'
NX_NODE_SIZE = 350
FLOAT_PRECISION = 3  # TODO: investigate SpiNNaker fixed-point arithmetic
ANNOTATION = 'Simulated with SpiNNaker_under_version(1!4.0.0-Riptalon)'
DEFAULT_SPYNNAKER_PARAMS = {
    'timestep': 1.,
    'time_scale_factor': 4
}


#
# Exposed functions
#

class PageRankSimulation:

    def __init__(self, run_time, edges, labels=None, parameters=None, damping=1):
        # Simulation parameters
        self._run_time     = run_time
        self._edges        = edges
        self._labels       = labels or self._gen_labels(self._edges)
        self._sim_vertices = self._gen_sim_vertices(self._labels)
        self._sim_edges    = self._gen_sim_edges(self._edges, self._labels, self._sim_vertices)
        self._parameters   = DEFAULT_SPYNNAKER_PARAMS
        self._parameters.update(parameters or {})
        # TODO: add support for Damping factor
        self._damping      = damping

        # Simulation state variables
        self._model = None
        self._sim_ranks = None
        self._input_graph = None

        # Numpy printing with some precision and no scientific notation
        np.set_printoptions(suppress=True, precision=FLOAT_PRECISION)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            p.end()
            return
        # else: exception is cascaded...

    #
    # Private functions, internal helpers
    #

    @staticmethod
    def _gen_labels(edges):
        return map(str, set([s for s, _ in edges] + [t for _, t in edges]))

    @staticmethod
    def _gen_sim_vertices(labels):
        return list(range(len(labels)))

    @staticmethod
    def _gen_sim_edges(edges, labels, sim_vertices):
        labels_to_ids = dict(zip(labels, sim_vertices))
        try:
            return [(labels_to_ids[src], labels_to_ids[tgt]) for src, tgt in edges]
        except KeyError:
            raise ValueError("Some 'edges' use nodes not defined in 'labels'")

    @staticmethod
    def _node_formatter(name):
        return "Node %s" % name

    @staticmethod
    def _float_formatter(number):
        return ("%.{}f".format(FLOAT_PRECISION)) % number

    def _get_ranks_string(self, ranks):
        """Pretty prints a table of ranks values

        :param ranks: dict of name-indexed rows of values, or list of a single row of values
        :return: None
        """
        # Multiple rows, indexed by row name
        table = PrettyTable([''] + map(self._node_formatter, self._labels))
        for name, row in ranks.items():
            table.add_row([name] + map(self._float_formatter, row))

        return table.get_string()

    def _check_sim_ran(self):
        """Raises an error is the simulation was not ran.

        State variable `self._model' serves as a proxy for determining if .start(...) was called.

        :return: None
        """
        if self._model is None:
            raise RuntimeError('You first need to .start(...) the simulation.')

    def _extract_sim_ranks(self):
        """Extracts the rank computed during the simulation.

        :return: np.array, ranks
        """
        self._check_sim_ran()

        if self._sim_ranks is None:
            self._sim_ranks = self._model.get_data(RANK).segments[0].filter(name=RANK)[0]
        return self._sim_ranks

    def _create_page_rank_model(self):
        """Maps the graph to sPyNNaker.

        :return: p.Population, the neural model to compute Page Rank
        """
        # Pre-processing, compute inbound / outbound edges for each node
        n_neurons = len(self._sim_vertices)
        outgoing_edges_count = [0] * n_neurons
        incoming_edges_count = [0] * n_neurons
        for src, tgt in self._sim_edges:
            outgoing_edges_count[src] += 1
            incoming_edges_count[tgt] += 1

        # Vertices
        pop = p.Population(
            n_neurons,
            Page_Rank(
                rank_init=1./n_neurons,
                incoming_edges_count=incoming_edges_count,
                outgoing_edges_count=outgoing_edges_count
            ), label="page_rank"
        )

        # Edges
        p.Projection(
            pop, pop,
            p.FromListConnector(self._sim_edges),
            synapse_type=SynapseDynamicsNoOp()
        )

        return pop

    def _verify_sim(self, get_string=False):
        """Verifies simulation results correctness.

        Checks the ranks results from the simulation match those given by a Python implementation of
        Page Rank in networkx.

        :return: bool, whether the results match
        """
        if self._input_graph is None:
            self.draw_input_graph(show_graph=False)

        msg = ""

        # Get last row of the ranks computed in the simulation
        computed_ranks = self._extract_sim_ranks()[-1]
        try:
            ranks_dict = nx.pagerank(self._input_graph, self._damping, weight=None)
            expected_ranks = np.array([ranks_dict[v] for v in self._labels])

            is_correct = np.allclose(computed_ranks, expected_ranks, atol=10**(-FLOAT_PRECISION))
        except nx.PowerIterationFailedConvergence:
            is_correct = True
            msg += "WARNING Page Rank python got PowerIterationFailedConvergence.\n"

        if is_correct:
            msg += "CORRECT Page Rank results.\n" \
                + self._get_ranks_string({
                    'Computed': computed_ranks
                })
        else:
            msg += "INCORRECT Page Rank results.\n" \
                + self._get_ranks_string({
                    'Computed': computed_ranks,
                    'Expected': expected_ranks
                })

        if get_string:
            return is_correct, msg

        print(msg)
        return is_correct

    #
    # Exposed functions
    #

    def run(self, verify=False, get_string=False):
        """Runs the simulation.

        :param verify: check the results with a Page Rank python implementation.
        :param get_string: get the table to results from verification
        :return: bool, correctness of the simulation results
        """
        # Setup simulation
        p.setup(**self._parameters)

        self._model = self._create_page_rank_model()
        self._model.record([RANK])

        p.run(self._run_time)

        if verify:
            return self._verify_sim(get_string)
        return True

    def draw_input_graph(self, show_graph=False):
        """Compute a graphical representation of the input graph.

        :param show_graph: whether to display the graph, default is False
        :return: None
        """
        # Clear plot
        plt.clf()

        # Graph structure
        G = nx.Graph().to_directed()
        G.add_edges_from(self._edges)

        # Graph layout
        pos = nx.layout.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=NX_NODE_SIZE, node_color='red')
        nx.draw_networkx_edges(G, pos, arrowstyle='->')
        nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold')
        self_loops = G.nodes_with_selfloops()
        nx.draw_networkx_nodes(self_loops, pos, node_size=NX_NODE_SIZE, node_color='black')

        # Save graph for Page Rank python computations
        self._input_graph = G

        # Show graph
        if show_graph:
            print("Displaying input graph. "
                  "Check DISPLAY={} if this hangs...".format(os.getenv('DISPLAY')))
            plt.gca().set_axis_off()
            plt.suptitle('Input graph for Page Rank')
            plt.title('Black nodes are self-looping', fontsize=8)
            plt.show()

    def draw_output_graph(self, show_graph=True, pause=False):
        """Displays the computed rank over time.

        Note: pausing the simulation before it ends and is unloaded from the SpiNNaker chips allows
        for inspection of the post-simulation state through `ybug'
        (see SpiNNakerManchester/spinnaker_tools)

        :param show_graph: whether to display the graph, default is False
        :param pause: whether to pause the simulation after showing results, default is False
        :return: None
        """
        self._check_sim_ran()

        ranks = self._extract_sim_ranks()
        time_step = (float(self._run_time) / len(ranks)) if len(ranks) != 0 else 0

        if show_graph:
            # Clear plot
            plt.clf()

            print("Displaying output graph. "
                  "Check DISPLAY={} if this hangs...".format(os.getenv('DISPLAY')))
            panel = Panel(
                ranks,
                ylabel="Rank", yticks=True,
                xlabel="Time (ms)", xticks=True, xlim=(0, self._run_time - time_step)
            )
            Figure(panel, title="Rank over time", annotations=ANNOTATION)

            # Override default legend
            texts = plt.gca().legend_.get_texts()
            labels = self._labels or []
            for t, label in zip(texts, labels):
                t.set_text(self._node_formatter(label))
            plt.show()

        if pause:
            raw_input('Press any key to finish...')


@contextmanager
def silence_stdout():
    new_target = open(os.devnull, "w")
    old_stdout, sys.stdout = sys.stdout, new_target
    old_stderr, sys.stderr = sys.stderr, new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

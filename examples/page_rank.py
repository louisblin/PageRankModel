import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from prettytable import PrettyTable

import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel

from python_models8.model_data_holders.page_rank_data_holder import PageRankDataHolder as Page_Rank
from python_models8.synapse_dynamics.synapse_dynamics_noop import SynapseDynamicsNoOp

RANK = 'v'
# TODO: investigate SpiNNaker fixed-point arithmetic
FLOAT_PRECISION = 3
GUI_TIMEOUT = 3000
NX_NODE_SIZE = 350
ANNOTATION = 'Simulated with SpiNNaker_under_version(1!4.0.0-Riptalon)'


#
# Exposed functions
#

class PageRankSimulation:

    def __init__(self, run_time, vertices, edges, parameters=None, labels=None, damping=1):
        # Simulation parameters
        self._run_time = run_time
        self._vertices = vertices
        self._edges = edges
        self._parameters = parameters
        self._labels = labels
        # TODO: add support for Damping factor
        self._damping = damping

        # Simulation state variables
        self._model = None
        self._sim_ranks  = None
        self._input_graph = None

        # Numpy printing with some precision and no scientific notation
        np.set_printoptions(suppress=True, precision=FLOAT_PRECISION)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        p.end()

    #
    # Private functions, internal helpers
    #

    @staticmethod
    def _node_formatter(name):
        return "Node %s" % name

    @staticmethod
    def _float_formatter(number):
        template = "%.{}f".format(FLOAT_PRECISION)
        return template % number

    def _print_ranks(self, ranks):
        """Pretty prints a table of ranks values

        :param ranks: dict of name-indexed rows of values, or list of a single row of values
        :return: None
        """
        lbl = self._labels or map(self._node_formatter, self._vertices)

        # If multiple rows, indexed by row name
        if isinstance(ranks, dict):
            table = PrettyTable([''] + lbl)
            for name, row in ranks.items():
                table.add_row([name] + map(self._float_formatter, row))
        # One line
        else:
            table = PrettyTable(lbl)
            table.add_row(map(self._float_formatter, ranks))
        print(table)

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
        n_neurons = len(self._vertices)
        outgoing_edges_count = [0] * n_neurons
        incoming_edges_count = [0] * n_neurons
        for src, tgt in self._edges:
            outgoing_edges_count[src] += 1
            incoming_edges_count[tgt] += 1

        # Vertices
        pop = p.Population(
            n_neurons,
            Page_Rank(
                rank_init= 1./n_neurons,
                incoming_edges_count=incoming_edges_count,
                outgoing_edges_count=outgoing_edges_count
            ), label="page_rank"
        )

        # Edges
        p.Projection(
            pop, pop,
            p.FromListConnector(self._edges),
            synapse_type=SynapseDynamicsNoOp()
        )

        return pop

    def _verify_sim(self):
        """Verifies simulation results correctness.

        Checks the ranks results from the simulation match those given by a Python implementation of
        Page Rank in networkx.

        :return: bool, whether the results match
        """
        if self._input_graph is None:
            self.draw_input_graph(show_graph=False)

        # Get last row of the ranks computed in the simulation
        computed_ranks = self._extract_sim_ranks()[-1]
        ranks_dict = nx.pagerank(self._input_graph, self._damping)
        expected_ranks = np.array([ranks_dict[v] for v in self._vertices])

        close = np.allclose(computed_ranks, expected_ranks, atol=10**(-FLOAT_PRECISION))

        if close:
            print("CORRECT Page Rank results.")
            self._print_ranks(computed_ranks)
        else:
            print("INCORRECT Page Rank results.")
            self._print_ranks({
                'Computed': computed_ranks,
                'Expected': expected_ranks
            })

        return close

    #
    # Exposed functions
    #

    def run(self, verify=False):
        """Runs the simulation.

        :param verify: check the results with a Page Rank python implementation.
        :return: bool, correctness of the simulation results
        """
        # Setup simulation
        p.setup(**self._parameters)

        self._model = self._create_page_rank_model()
        self._model.record([RANK])

        p.run(self._run_time)

        if verify:
            return self._verify_sim()
        return True

    def draw_input_graph(self, show_graph=False):
        """Compute a graphical representation of the input graph.

        :param show_graph: whether to display the graph, default is False
        :return: None
        """
        # Labels in the legend
        labels = None
        if self._labels:
            labels = dict(zip(self._vertices, self._labels))

        # Graph structure
        G = nx.Graph().to_directed()
        G.add_edges_from(self._edges)

        # Graph layout
        pos = nx.layout.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=NX_NODE_SIZE, node_color='red')
        nx.draw_networkx_edges(G, pos, arrowstyle='->')
        nx.draw_networkx_labels(G, pos, labels=labels, font_color='white', font_weight='bold')
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
        time_step = (float(self._run_time ) / len(ranks)) if len(ranks) != 0 else 0

        if show_graph:
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
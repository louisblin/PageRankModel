import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import csc_matrix

from pyNN.utility.plotting import Figure, Panel

from python_models8.model_data_holders.page_rank_data_holder import PageRankDataHolder as Page_Rank
from python_models8.synapse_dynamics.synapse_dynamics_noop import SynapseDynamicsNoOp

ANNOTATION = 'Simulated with SpiNNaker_under_version(1!4.0.0-Riptalon)'

# Numpy printing with some precision and no scientific notation
np.set_printoptions(suppress=True, precision=5)


def create_page_rank_model(p, vertices, edges):

    # Compute number of inbound edges for each vertex
    n_neurons = len(vertices)
    outgoing_edges_count = [0] * n_neurons
    incoming_edges_count = [0] * n_neurons
    for src, tgt in edges:
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
        p.FromListConnector(edges),
        synapse_type=SynapseDynamicsNoOp()
    )

    return pop


def draw_input_graph(edges, node_size=350):
    # Graph structure
    G = nx.generators.directed.random_k_out_graph(0, 0, 0)
    G.add_edges_from(edges)

    # Graph layout
    pos = nx.layout.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='red')
    nx.draw_networkx_edges(G, pos, arrowstyle='->')
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold')
    self_loops = G.nodes_with_selfloops()
    nx.draw_networkx_nodes(self_loops, pos, node_size=node_size, node_color='black')

    # Show graph
    plt.gca().set_axis_off()
    plt.suptitle('Input graph for Page Rank')
    plt.title('Black nodes are self-looping', fontsize=8)
    plt.show()


def draw_output_graph(model, y, run_time, show_graph=True):

    rank = model.get_data(y).segments[0].filter(name=y)[0]
    time_step = (float(run_time) / len(rank)) if len(rank) != 0 else 0

    print('===============================')
    print("Recorded values for '{}'\n{}".format(model.label, rank))
    print('===============================')

    if show_graph:
        panel = Panel(
            rank, data_labels=[model.label],
            ylabel="Rank", yticks=True,
            xlabel="Time (ms)", xticks=True, xlim=(0, run_time - time_step)
        )
        Figure(panel, title="Rank over time", annotations=ANNOTATION)
        plt.show()


def _page_rank(G, s=1., maxerr=.001):
    """
    Computes the pagerank for each of the n states
    Parameters
    ----------
    G:      binary-valued matrix where Gij represents a transition from state i to j.
    s:      probability of following a transition. 1-s probability of teleporting to another state.
    maxerr: if the sum of pageranks between iterations is bellow this we will have converged.
    """
    n = G.shape[0]

    # transform G into markov matrix A
    A = csc_matrix(G, dtype=np.float)
    rsums = np.array(A.sum(1))[:,0]
    ri, ci = A.nonzero()
    A.data /= rsums[ri]

    # bool array of sink states
    sink = rsums==0

    # Compute pagerank r until we converge
    ro, r = np.zeros(n), np.ones(n)
    while np.sum(np.abs(r-ro)) > maxerr:
        ro = r.copy()
        # calculate each pagerank at a time
        for i in xrange(0, n):
            # inlinks of state i
            Ai = np.array(A[:,i].todense())[:,0]
            # account for sink states
            Di = sink / float(n)
            # account for teleportation to state i
            Ei = np.ones(n) / float(n)

            r[i] = ro.dot( Ai*s + Di*s + Ei*(1-s) )

    # return normalized pagerank
    return r / float(sum(r))


def python_page_rank(vertices, edges, show=False):
    n_neurons = len(vertices)

    G = np.zeros((n_neurons , n_neurons))
    for src, tgt in edges:
        G[src][tgt] = 1

    pr = _page_rank(G)

    if show:
        print('===============================')
        print("Expected values for 'page_rank'\n{}".format(pr))
        print('===============================')

    return pr

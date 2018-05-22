import matplotlib.pyplot as plt
import networkx as nx

from pyNN.utility.plotting import Figure, Panel

ANNOTATION = 'Simulated with SpiNNaker_under_version(1!4.0.0-Riptalon)'


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


def _extract_value(m, y):
    return m.get_data(y).segments[0].filter(name=y)[0]


def draw_output_graph(models, y, run_time):

    panels = []
    for m in models:
        rank = _extract_value(m, y)
        time_step = (float(run_time) / len(rank)) if len(rank) != 0 else 0

        panels.append(Panel(
            rank, data_labels=[m.label],
            ylabel="Rank", yticks=True,
            xlabel="Time (ms)", xticks=True, xlim=(0, run_time - time_step)
        ))

        print('===============================')
        print("Recorded values for '{}'\n{}".format(m.label, rank))
        print('===============================')

    Figure(*panels, title="Rank over time", annotations=ANNOTATION)
    plt.show()

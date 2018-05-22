import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

from python_models8.model_data_holders.page_rank_data_holder import PageRankDataHolder as Page_Rank
from python_models8.synapse_dynamics.synapse_dynamics_noop import SynapseDynamicsNoOp

import sys


RANK = 'v'

###############################################################################
# Simulation parameters

run_time = 100
time_step = 1.0
n_neurons = 5

p.setup(time_step, min_delay=time_step)

###############################################################################
# Construct simulation graph

# Connect graph in a ring
injectionConnections = []
incoming_edges_count = [0] * n_neurons
for src in range(n_neurons):
    tgt = (src + 1) % n_neurons
    injectionConnections.append((src, tgt))
    incoming_edges_count[tgt] += 1

pop = p.Population(
    n_neurons,
    Page_Rank(incoming_edges_count=incoming_edges_count),
    label="page_rank"
)
p.Projection(
    pop, pop,
    p.FromListConnector(injectionConnections),
    synapse_type=SynapseDynamicsNoOp()
)
pop.record([RANK])

models = [pop]

###############################################################################
# Run simulation / report

p.run(run_time)

# Graph reporting
panels = []
for m in models:
    rank = m.get_data(RANK).segments[0].filter(name=RANK)[0]
    panel = Panel(rank, ylabel="Rank (au)", data_labels=[m.label], yticks=True, xlim=(0, run_time))
    panels.append(panel)
    print('label={} :: {}'.format(m.label, rank))

# Optionally skip report
if len(sys.argv) >= 2 and sys.argv[1] == '--pause':
    raw_input('Press any key to continue...')
else:
    Figure(*panels, title="Rank over time", annotations="Simulated with {}".format(p.name()))
    plt.show()

p.end()
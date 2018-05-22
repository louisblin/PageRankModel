import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

from python_models8.model_data_holders.page_rank_data_holder import PageRankDataHolder as Page_Rank
from python_models8.synapse_dynamics.synapse_dynamics_noop import SynapseDynamicsNoOp

import sys


# Set the run time of the execution
run_time = 100

# Set the time step of the simulation in milliseconds
time_step = 1.0

# Set the number of neurons to simulate
n_neurons = 5

# Set the times at which to input a spike
spike_times = range(0, run_time, 20)

p.setup(time_step, min_delay=time_step)

# spikeArray = {
#     "spike_times": spike_times
# }
# input_pop = p.Population(n_neurons, p.SpikeSourceArray(**spikeArray), label="input")


models = []
def synapsify(my_model):

    # p.Projection(input_pop, my_model,
    #              p.OneToOneConnector(),
    #              receptor_type='excitatory', synapse_type=p.StaticSynapse(weight=weight))
    connections = [
        (0, 1),
        (1, 2),
        (2, 0),
    ]

    p.Projection(my_model, my_model,
                 p.FromListConnector(connections),
                 synapse_type=SynapseDynamicsNoOp())

    my_model.record(['v'])

    models.append(my_model)


# Connected models
page_rank_parameters = {
    'incoming_edges_count': 1
}

for i in range(1):
    synapsify(p.Population(n_neurons, Page_Rank(**page_rank_parameters),
                           label="page_rank_{}".format(i+1)))

# Run
p.run(run_time)

# Report
panels = []
for m in models:
    v = m.get_data('v').segments[0].filter(name='v')[0]
    # membrane potentials for each example
    panels.append(Panel(
        v, ylabel="Membrane potential (mV)",
        data_labels=[m.label], yticks=True, xlim=(0, run_time))
    )
    print('label={} :: {}'.format(m.label, v))

if len(sys.argv) >= 2 and sys.argv[1] == '--pause':
    raw_input('Press any key to continue...')
else:
    Figure(*panels, title="Custom models", annotations="Simulated with {}".format(p.name()))
    plt.show()

p.end()

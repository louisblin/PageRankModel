# import spynnaker8 and plotting stuff
import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

import sys

from python_models8.model_data_holders.page_rank_data_holder \
    import PageRankDataHolder as Page_Rank


# Set the run time of the execution
run_time = 300

# Set the time step of the simulation in milliseconds
time_step = 1.0

# Set the number of neurons to simulate
n_neurons = 3

# Set the weight of input spikes
weight = 10.0

# Set the times at which to input a spike
spike_times = range(0, run_time, 100)

p.setup(time_step)

spikeArray = {
    "spike_times": spike_times
}
input_pop = p.Population(n_neurons, p.SpikeSourceArray(**spikeArray), label="input")


models = []
def synapsify(my_model):
    p.Projection(input_pop, my_model,
                 p.OneToOneConnector(),
                 receptor_type='excitatory', synapse_type=p.StaticSynapse(weight=weight))

    my_model.record(['v'])

    models.append(my_model)


# Connected models
for i in range(1):
    synapsify(p.Population(n_neurons, Page_Rank(v_thresh=1), label="page_rank_{}".format(i+1)))

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

if len(sys.argv) >=2 and sys.argv[1] == '--pause':
    raw_input('Press any key to continue...')
else:
    Figure(*panels, title="Custom models", annotations="Simulated with {}".format(p.name()))
    plt.show()

p.end()

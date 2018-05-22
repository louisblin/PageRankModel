from spynnaker.pyNN.models.neuron.synapse_dynamics \
    import SynapseDynamicsStatic as CommonSynapseDynamicsStatic


class SynapseDynamicsNoOp(CommonSynapseDynamicsStatic):
    def __init__(self, *args, **kwargs):
        CommonSynapseDynamicsStatic.__init__(self)

    @property
    def weight(self):
        return 1

    @weight.setter
    def weight(self, new_value):
        pass

    @property
    def delay(self):
        return 1

    @delay.setter
    def delay(self, new_value):
        pass

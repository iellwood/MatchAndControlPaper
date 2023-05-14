"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

A dendrite covered in spines
"""


from Models import dendrite, spine
from neuron.units import ms, mV
import numpy as np
import Models.basic_properties as basic_properties

class SpinyDendriteModel:

    def __init__(self, h, number_of_spines=1, include_ca_dependent_potentiation=False, dendrite_parameters=None):
        self.h = h
        if dendrite_parameters is None:
            dendrite_parameters = basic_properties.dendrite_parameters_for_dendrite_only_model


        self.distal_dendrite = dendrite.Dendrite(h, dendrite_parameters, name='dendrite')

        self.distal_dendrite.add_mechanisms(basic_properties.mechanism_properties_list)

        self.spine_list = [spine.Spine(h, basic_properties.spine_parameter_dict) for i in range(number_of_spines)]
        self.ampa_list = []
        self.nmda_list = []
        self.vec_stims = []
        self.netcons_ampa = []
        self.netcons_nmda = []

        for s in self.spine_list:
            s.add_spine_mechanism(basic_properties.passive_membrane_properties)
            s.head.insert('cadspine')
            if include_ca_dependent_potentiation:
                self.ampa_list.append(h.AMPA_WITH_CA(s.head(0.5)))

            else:
                self.ampa_list.append(h.AMPA_WITH_CA_NO_POTENTIATION(s.head(0.5)))
            self.nmda_list.append(h.NMDA(s.head(0.5)))

            self.vec_stims.append(h.VecStim())
            self.netcons_ampa.append(h.NetCon(self.vec_stims[-1], self.ampa_list[-1]))
            self.netcons_nmda.append(h.NetCon(self.vec_stims[-1], self.nmda_list[-1]))

        for n in self.netcons_nmda:
            n.weight[0] = 0.003

        for n in self.netcons_ampa:
            n.weight[0] = 0.003

        locations = np.linspace(dendrite_parameters['proximal_padding'], dendrite_parameters['length'] - dendrite_parameters['distal_padding'], number_of_spines)
        locations = locations/dendrite_parameters['length']

        for i, s in enumerate(self.spine_list):
            self.distal_dendrite.add_spine(s, locations[i])

        self.t = h.Vector().record(h._ref_t)

        self.cas = []
        self.vs = []
        for s in self.spine_list:
            self.vs.append(self.h.Vector().record(s.head(0.5)._ref_v))
            self.cas.append(self.h.Vector().record(s.head(0.5)._ref_cai))

        self.gs = []
        for ampa_channel in self.ampa_list:
            self.gs.append(self.h.Vector().record(ampa_channel._ref_g_dynamic))

        self.iclamps = []

        self.distal_dendrite.section.ek = basic_properties.dendrite_parameters_for_dendrite_only_model['ek']

    def set_AMPA_weight(self, spine_number, weight):
        if type(spine_number) is int:
            self.netcons_ampa[spine_number].weight[0] = weight
        elif type(spine_number) is list:
            for i in spine_number:
                self.netcons_ampa[i].weight[0] = weight
        elif spine_number == 'all':
            for netcon in self.netcons_ampa:
                netcon.weight[0] = weight
        else:
            raise Exception("spine_number of unknown type. Must be int, list of ints or 'all'.")

    def set_NMDA_weight(self, spine_number, weight):
        if type(spine_number) is int:
            self.netcons_nmda[spine_number].weight[0] = weight
        elif type(spine_number) is list:
            for i in spine_number:
                self.netcons_nmda[i].weight[0] = weight
        elif spine_number == 'all':
            for netcon in self.netcons_nmda:
                netcon.weight[0] = weight
        else:
            raise Exception("spine_number of unknown type. Must be int, list of ints or 'all'.")


    def add_baps(self, stimulation_times_ms, stimulation_duration_ms, amplitude_nA):
        self.iclamps = []
        for i in range(len(stimulation_times_ms)):
            stimobj = self.h.IClamp(self.distal_dendrite.section(0.0))
            stimobj.delay = stimulation_times_ms[i]
            stimobj.amp = amplitude_nA
            stimobj.dur = stimulation_duration_ms
            self.iclamps.append(stimobj)

        # for i in range(len(stimulation_times_ms)):
        #     HocPythonTools.add_IClamp_to_section(self.h, 'current_pulse_' + str(i), self.distal_dendrite.section, stimulation_times_ms[i], stimulation_duration_ms, amplitude_nA, location=0)

    def add_presynaptic_stimulation(self, stimuli_time_array_ms):
        for i in range(len(self.vec_stims)):
            self.vec_stims[i].play(self.h.Vector(stimuli_time_array_ms[i]))

    def record(self, duration_ms):

        self.h.finitialize(-70.8341 * mV)
        self.h.continuerun(duration_ms * ms)

        self.iclamps = []


        return np.array(self.t), np.array(self.vs), np.array(self.cas)











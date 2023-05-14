"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

The complete pyramidal neuron model
"""


from Models.bahletal2012model import L5NeuronReducedModel_BahlEtAl2012
from Models.spinydendritemodel import SpinyDendriteModel
import Models.basic_properties as basic_properties
from Models.dendrite import Dendrite
import numpy as np

class Neuron:

    def __init__(self, h, include_ca_dependent_potentiation=False, tuft_dendrite_branch_length=250, distance_between_spines=1):

        self.h = h

        self.number_of_spines_per_dendrite = 250
        self.synapses_per_axon = 10
        self.number_of_spines = self.number_of_spines_per_dendrite * 6
        self.dendrite_length = tuft_dendrite_branch_length

        basic_properties.passive_membrane_properties['e'] = -85
        basic_properties.passive_membrane_properties['g'] = 0.0002

        dendrite_parameters_thin = {
            'dendrite diameter': 0.5,
            'length': self.dendrite_length,
            'proximal_padding': 2,  # amount of dendrite to include with no spines on the proximal side
            'distal_padding': 2,  # amount of dendrite to include with no spines on the distal side
            'cm': basic_properties.membrane_capacitance,
            'Ra': 100,
            'ek': -85,
        }

        dendrite_parameters_medium = {
            'dendrite diameter': 1,
            'length': 50,
            'proximal_padding': 0,  # amount of dendrite to include with no spines on the proximal side
            'distal_padding': 0,  # amount of dendrite to include with no spines on the distal side
            'cm': basic_properties.membrane_capacitance,
            'Ra': 100,
            'ek': -85,
        }

        dendrite_parameters_thick = {
            'dendrite diameter': 2,  # The diameters get overwritten below using a smoothed version of Hay et. al.'s apical dendrite diameters
            'length': 450,
            'proximal_padding': 0,  # amount of dendrite to include with no spines on the proximal side
            'distal_padding': 0,  # amount of dendrite to include with no spines on the distal side
            'cm': basic_properties.membrane_capacitance,
            'Ra': 100,
            'ek': -85,
        }

        self.axosomatic_compartments = L5NeuronReducedModel_BahlEtAl2012(h, include_apical=False)

        # Create the apical dendrite sections
        def make_dendrite_branch():
            return SpinyDendriteModel(
                self.h, number_of_spines=self.number_of_spines_per_dendrite,
                include_ca_dependent_potentiation=include_ca_dependent_potentiation,
                dendrite_parameters=dendrite_parameters_thin
            )


        self.spiny_dendrite_0 = make_dendrite_branch()
        self.spiny_dendrite_0_0 = make_dendrite_branch()
        self.spiny_dendrite_0_1 = make_dendrite_branch()

        self.spiny_dendrite_1 = make_dendrite_branch()
        self.spiny_dendrite_1_0 = make_dendrite_branch()
        self.spiny_dendrite_1_1 = make_dendrite_branch()

        self.spiny_branches = [
            self.spiny_dendrite_0,
            self.spiny_dendrite_1,
            self.spiny_dendrite_0_0,
            self.spiny_dendrite_0_1,
            self.spiny_dendrite_1_0,
            self.spiny_dendrite_1_1,
        ]

        self.apical_dendrite_shaft = Dendrite(h, dendrite_parameters_thick)
        self.apical_dendrite_shaft_top = Dendrite(h, dendrite_parameters_medium)


        # Add apical dendrite main shaft ion mechanisms
        self.apical_dendrite_shaft.add_mechanisms(basic_properties.mechanism_properties_list)
        self.apical_dendrite_shaft_top.add_mechanisms(basic_properties.mechanism_properties_list)


        # Connect the segments together:

        # Apical to soma
        self.apical_dendrite_shaft.section.connect(self.axosomatic_compartments.soma(1))
        self.apical_dendrite_shaft_top.section.connect(self.apical_dendrite_shaft.section(1))

        # first two tuft branches to apical
        self.spiny_dendrite_0.distal_dendrite.section.connect(self.apical_dendrite_shaft_top.section(1))
        self.spiny_dendrite_1.distal_dendrite.section.connect(self.apical_dendrite_shaft_top.section(1))

        # branches of first tuft branch to first tuft branch
        self.spiny_dendrite_0_0.distal_dendrite.section.connect(self.spiny_dendrite_0.distal_dendrite.section(1))
        self.spiny_dendrite_0_1.distal_dendrite.section.connect(self.spiny_dendrite_0.distal_dendrite.section(1))

        # branches of second tuft branch to second tuft branch
        self.spiny_dendrite_1_0.distal_dendrite.section.connect(self.spiny_dendrite_1.distal_dendrite.section(1))
        self.spiny_dendrite_1_1.distal_dendrite.section.connect(self.spiny_dendrite_1.distal_dendrite.section(1))

        for seg in self.apical_dendrite_shaft.section:
            seg.pas.e = -75
            seg.NaTa_t.gNaTa_tbar = seg.NaTa_t.gNaTa_tbar*6
            seg.SKv3_1.gSKv3_1bar = seg.SKv3_1.gSKv3_1bar*6
            seg.diam = basic_properties.apical_diameter_model(h.distance(seg))
            last_diam = seg.diam

        for seg in self.axosomatic_compartments.soma:
            seg.pas.e = -75

        distances = []
        self.apical_dendrite_shaft_top.section.Ra = 500
        for s in self.spiny_branches:
            s.distal_dendrite.section.Ra = 200
        for seg in self.apical_dendrite_shaft_top.section:
            seg.pas.e = -75
            seg.pas.g = seg.pas.g
            seg.NaTa_t.gNaTa_tbar = seg.NaTa_t.gNaTa_tbar*8
            seg.SKv3_1.gSKv3_1bar = seg.SKv3_1.gSKv3_1bar*8
            distances.append(self.h.distance(seg))

        diams = np.geomspace(last_diam, 0.5, len(distances))
        for i, seg in enumerate(self.apical_dendrite_shaft_top.section):
            seg.diam = diams[i]

        for spiny_dendrite in self.spiny_branches:
            for seg in spiny_dendrite.distal_dendrite.section:
                seg.NaTa_t.gNaTa_tbar = seg.NaTa_t.gNaTa_tbar*2
                seg.SKv3_1.gSKv3_1bar = seg.SKv3_1.gSKv3_1bar*1.5

        # for spiny_dendrite in self.spiny_branches[:2]:
        #     for seg in spiny_dendrite.distal_dendrite.section:
        #         seg.NaTa_t.gNaTa_tbar = seg.NaTa_t.gNaTa_tbar*2.5
        #         seg.SKv3_1.gSKv3_1bar = seg.SKv3_1.gSKv3_1bar*2.5 * 1.5 / 2

        # for seg in self.spiny_dendrite_0.distal_dendrite.section:
        #     seg.pas.e = -92
        # for seg in self.spiny_dendrite_1.distal_dendrite.section:
        #     seg.pas.e = -92


        self.v_soma = h.Vector().record(self.axosomatic_compartments.soma(0.5)._ref_v)
        self.v_apical_dendrite_shaft = [h.Vector().record(segment._ref_v) for segment in self.apical_dendrite_shaft.section]

        self.v_spine = []
        self.ca_spine = []
        self.g_spine = []
        for branch in self.spiny_branches:
            self.v_spine.extend(branch.vs)
            self.ca_spine.extend(branch.cas)
            self.g_spine.extend(branch.gs)

        self.set_branch_spine_ID_vectors()

    def add_presynaptic_stimulation(self, stimuli_time_array_ms):
        spine_number_offset = 0
        for branch in self.spiny_branches:
            n = len(branch.spine_list)
            branch.add_presynaptic_stimulation(stimuli_time_array_ms[spine_number_offset:spine_number_offset + n])
            spine_number_offset += n

    def set_branch_spine_ID_vectors(self):
        self.branch_ID = []
        self.branch_spine_number = []
        for i, branch in enumerate(self.spiny_branches):
            for spine_number in range(len(branch.spine_list)):
                self.branch_ID.append(i)
                self.branch_spine_number.append(spine_number)

        self.setup_AMPA_and_NMDA()

    def set_AMPA_weight(self, spine_number, weight):
        if type(spine_number) is int:
            self.spiny_branches[self.branch_ID[spine_number]].set_AMPA_weight(self.branch_spine_number[spine_number], weight)
        elif type(spine_number) is list or type(spine_number) is np.array:
            for i in spine_number:
                self.spiny_branches[self.branch_ID[i]].set_AMPA_weight(self.branch_spine_number[i], weight)
        elif spine_number == 'all':
            for i in range(self.number_of_spines):
                self.spiny_branches[self.branch_ID[i]].set_AMPA_weight(self.branch_spine_number[i], weight)
        else:
            raise Exception("spine_number of unknown type. Must be int, list of ints or 'all'.")

    def set_NMDA_weight(self, spine_number, weight):
        if type(spine_number) is int:
            self.spiny_branches[self.branch_ID[spine_number]].set_NMDA_weight(self.branch_spine_number[spine_number], weight)
        elif type(spine_number) is list or type(spine_number) is np.array:
            for i in spine_number:
                self.spiny_branches[self.branch_ID[i]].set_NMDA_weight(self.branch_spine_number[i], weight)
        elif spine_number == 'all':
            for i in range(self.number_of_spines):
                self.spiny_branches[self.branch_ID[i]].set_NMDA_weight(self.branch_spine_number[i], weight)
        else:
            raise Exception("spine_number of unknown type. Must be int, list of ints or 'all'.")

    def setup_AMPA_and_NMDA(self, special_spines=None):

        scale = 0.08
        self.set_AMPA_weight('all', 0.1 * scale)
        self.set_NMDA_weight('all', 0.00005 * scale)

        if special_spines is not None:
            for i in range(len(special_spines)):
                self.set_AMPA_weight(special_spines[i], 0.8 * scale)

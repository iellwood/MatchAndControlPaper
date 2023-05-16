"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

This script prints out the parameters of the model, including the all geometric, electrical,
and ion channel conductances.
"""

import HocPythonTools
from Models.completeneuron import Neuron

# Set up the hoc interpreter and load the mechanisms
h = HocPythonTools.setup_neuron('ChannelModFiles/x86_64/.libs/libnrnmech.so')
model = Neuron(h, include_ca_dependent_potentiation=True)

AMPA_base_g = model.spiny_branches[0].ampa_list[0].gmax
AMPA_netcon_weight = model.spiny_branches[0].netcons_ampa[0].weight[0]
AMPA_conductance = AMPA_base_g * AMPA_netcon_weight

NMDA_base_g = model.spiny_branches[0].nmda_list[0].gmax
NMDA_netcon_weight = model.spiny_branches[0].netcons_nmda[0].weight[0]
NMDA_conductance = NMDA_base_g * NMDA_netcon_weight

# Make a list of all the sections whose parameters will be printed
all_sections = model.axosomatic_compartments.all_sections
all_sections.append(model.apical_dendrite_shaft.section)
all_sections.append(model.apical_dendrite_shaft_top.section)
all_sections.append(model.spiny_branches[0].distal_dendrite.section)

# Make a list of the names of each segment for human readability
all_sections_names = model.axosomatic_compartments.all_section_names
all_sections_names.append('Apical shaft')
all_sections_names.append('Apical shaft top')
all_sections_names.append('Spiny dendrite in apical tuft')

# A list of all the channels whose properties will be printed
channel_list = [
    'BahlEtAl_2012_cad',
    'BahlEtAl_2012_ih',
    'BahlEtAl_2012_km',
    'BahlEtAl_2012_kca',
    'BahlEtAl_2012_kfast',
    'BahlEtAl_2012_kslow',
    'BahlEtAl_2012_nap',
    'BahlEtAl_2012_nat',
    'BahlEtAl_2012_sca',
    'AMPA_WITH_CA',
    'AMPA_WITH_CA_NO_POTENTIATION',
    'cadspine',
    'Im',
    'NaTa_t',
    'NMDA',
    'SKv3_1',
]
print('')
print('Model properties from I. T. Ellwood, "Short-term Hebbian learning can implement transformer-like attention".')
print('')
print('Notes:')
print('\tWhen diam is a list, the segment is tapered. Otherwise the diameter is constant.')
print('\tIon channel properties that are zero are not printed.')
print('\tIon channels in the axon change along the segment, but only the value in the first segment is printed.')
print('\tOnly one of the dendritic tuft branches is shown. The others are identical.')
print('')

print('Spine ion channel conductances')
print('\tSingle spine total AMPA conductance (nS) =', AMPA_conductance * 1000)
print('\tSingle spine total AMPA conductance Potentiated (nS) =', AMPA_conductance * 8 * 1000)
print('\tSingle spine total NMDA conductance (nS) = {g:.4f}'.format(g=NMDA_conductance * 1000))
print('\t(See paper for discussion of possible mechanisms for AMPA component of potentiation increase.)')
print('')


HocPythonTools.print_model_parameters(h, all_sections, all_sections_names, channel_list)


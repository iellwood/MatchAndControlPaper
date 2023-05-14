"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

Some basic properties used in building the neuron model.

WARNING: Many of these parameters are overwritten by the model. For a printout of all the parameter
models, please see the main README file for instructions
"""



import numpy as np

# Parameters from Hay et. al.

# Conductances:
gSK_E2bar_SK_E2   = 0.0012
gSKv3_1bar_SKv3_1 = 0.000261
gNaTa_tbar_NaTa_t = 0.0213
gImbar_Im         = 0.0000675
g_pas             = 0.0000589

# Reversal potentials
e_pas             = -90

# Modified parameters

no_spine_area = ((0.5/2)**2)*1
spine_area = ((0.5/2)**2)*0.5 + (0.1/2)**2 * 1

capacitance_rescale = (no_spine_area * 2 + spine_area * 1)/(no_spine_area + spine_area)


membrane_capacitance = 1

#s = 1.5792  # Increase in surface area from adding spines

e_pas = -80   # - 70 for 50 spines
g_pas = 0.0001 * (no_spine_area) / (spine_area + no_spine_area) * 2.5 # This factor of 2.5 is only needed for the full model
# print('g_pas =', g_pas)
gSKv3_1bar_SKv3_1 = 0.04  # Old values 0.04
gNaTa_tbar_NaTa_t = 0.05  # Old values 0.05

BahlEtAl2012_pas = {
    'mod': 'pas',
    'g': 1.0/15000,
    'e': -70,
}

BahlEtAl2012_ih = {
    'mod': 'BahlEtAl_2012_ih',
    'ehd': -47,
}

BahlEtAl2012_nat = {
    'mod': 'BahlEtAl_2012_nat',
    'ena': 55,
    'vshift': 10,
}

BahlEtAl2012_kfast = {
    'mod': 'BahlEtAl_2012_kfast',
    'ek': -80
}

BahlEtAl2012_kslow = {
    'mod': 'BahlEtAl_2012_kslow',
    'ek': -80
}

BahlEtAl2012_cad = {
    'mod': 'BahlEtAl_2012_cad',
    'eca': 140,
}

BahlEtAl2012_nap = {
    'mod': 'BahlEtAl_2012_nap'
}

BahlEtAl2012_km = {
    'mod': 'BahlEtAl_2012_km',
    'gbar': 0,
}

BahlEtAl2012_sca = {
    'mod': 'BahlEtAl_2012_sca'
}

BahlEtAl2012_kca = {
    'mod': 'BahlEtAl_2012_kca'
}



dendrite_parameters_for_dendrite_only_model = {
    'dendrite diameter': 0.5,
    'length': 1000,
    'proximal_padding': 250, # amount of dendrite to include with no spines on the proximal side
    'distal_padding': 250,   # amount of dendrite to include with no spines on the distal side
    'cm': membrane_capacitance,
    'Ra': 100,
    'ek': -85,
}

dendrite_parameters_for_full_model = {
    'dendrite diameter': 0.5,
    'length': 520,
    'proximal_padding': 2,  # amount of dendrite to include with no spines on the proximal side
    'distal_padding': 2,    # amount of dendrite to include with no spines on the distal side
    'cm': membrane_capacitance,
    'Ra': 100,
    'ek': -85,
}

spine_parameter_dict = {
    'head_length': 0.5,  # Spine model from Jack et. al 1989, Major et. al. 1994, Badoual et. al. 2006
    'head_diameter': 0.5,
    'neck_length': 1.0,
    'neck_diameter': 0.1,
    'cm': membrane_capacitance,
    'Ra': 100,
}

passive_membrane_properties = {
    'mod': 'pas',
    'e': e_pas,
    'g': g_pas
}

fast_sodium_properties = {
    'mod': 'NaTa_t',
    'gNaTa_tbar': gNaTa_tbar_NaTa_t,
}

fast_potassium_properties = {
    'mod': 'SKv3_1',
    'gSKv3_1bar': gSKv3_1bar_SKv3_1,
#    'e_k': -85
}

cadspine_properties = {
    'mod': 'cadspine',
    'lneck_cadspine': spine_parameter_dict['neck_length'],
    'sneck_cadspine': np.pi * (spine_parameter_dict['neck_diameter'] ** 2) / 4,
    'vspine_cadspine': np.pi * (spine_parameter_dict['head_diameter'] ** 2) * spine_parameter_dict['head_length'] / 4,
}

nmda_properties = {
    'mod': 'NMDA',
    'gmax': 0.05,
}

ampa_properties = {
    'mod': 'AMPA',
    'g': 0.002,
}


Im_properties = {
    'mod': 'Im',
    'gImbar': gImbar_Im,
}

mechanism_properties_list = [passive_membrane_properties, fast_sodium_properties, fast_potassium_properties, Im_properties]


# A fit to the diameters in Hay et. al. 2011 using a model diameter = A + B * softplus(-distance/tau)
apical_diameter_model_params = [2.7642336, 21.55532, 15.393625]

def softplus_numpy(x):
    return np.where(x > 0, x + np.log(1 + np.exp(-x)), np.log(1 + np.exp(x)))

def apical_diameter_model(distance):
    return apical_diameter_model_params[0] + apical_diameter_model_params[1]*softplus_numpy(-distance/apical_diameter_model_params[2])

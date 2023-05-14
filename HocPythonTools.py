"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

Misc. helper functions for interfacing with NEURON.
"""

import neuron
import numpy as np
import matplotlib.pyplot as plt

def add_voltage_recording_to_section(h, section, recording_variable_name, location=None):
    '''
    Returns a reference to a vector that will contain a voltage recording of the specified section

    :param h: HocObject
    :param section: Section of the cell to be recorded (object reference).
    :param recording_variable_name: Name of the new recording variable to store the recording
    :return: Reference to the recording object
    '''
    # objref recording_variable_name
    # recording_variable_name = new Vector()
    # access 5PC.apic[siteVec[0]]
    # cvode.record( & v(siteVec[1]), recording_variable_name, tvec)

    v_vec = h.Vector()
    if location is None: location = 0.5
    v_vec.record(section(location)._ref_v)
    return v_vec

def add_mechanism(h, section, mod_dict):
    '''
    Adds a mechanism to the section.

    :param h: HocObject
    :param section: Section to add the mechanism to
    :param mod_dict: A dictionary of parameters to set. Must include the key 'mod' with value equal to the name of the mechanism.
    Example: {'mod': 'pas', 'g': 0.001}
    '''
    section.insert(mod_dict['mod'])

    for key in mod_dict.keys():
        if key != 'mod' and key != 'ena' and key != 'ek' and key != 'eca' and key != 'ecl':
            for seg in section:
                setattr(getattr(seg, mod_dict['mod']), key, mod_dict[key])
        elif key == 'ena':
            section.ena = mod_dict['ena']
        elif key == 'ek':
            section.ek = mod_dict['ek']
        elif key == 'eca':
            section.eca = mod_dict['eca']
        elif key == 'ecl':
            section.ecl = mod_dict['ecl']

def add_ca_recording_to_section(h, section, recording_variable_name):
    '''
    Executes the hoc commands

    >> objref recording_variable_name

    >> recording_variable_name = new Vector()

    >> recording_variable_name.record(&section_name.cai(0.5))
    Returns the recording_variable_name reference. Note that you must run the stimulation
    before the variable will contain a recording.

    :param h: HocObject
    :param section: Section of the cell to be recorded (object reference).
    :param recording_variable_name: Name of the new recording variable to store the recording
    :return: Reference to the recording object
    '''

    h('objref ' + recording_variable_name)
    h(recording_variable_name + ' = new Vector()')
    h(recording_variable_name + '.record(&' + section.name() + '.cai(0.5))')

    return getattr(h, recording_variable_name)


def plot_graph(sections, axis=None, values_normalized=None, color=None):
    for i, d in enumerate(sections):
        n = d.n3d()
        cmap = plt.get_cmap('plasma')
        if values_normalized is not None:
            color = cmap(values_normalized[i])
        else:
            color = color
        if axis is None:
            axis = plt.gca()
        if n > 0:
            for j in range(n - 1):
                axis.plot([d.x3d(j), d.x3d(j + 1)], [d.y3d(j), d.y3d(j + 1)], color=color)
    axis.axis('off')
    axis.set_aspect(1)


def get_section_centers(h, sections):
    centers = []
    for s in sections:
        n = s.n3d()
        if n > 0:
            q = np.zeros((3,), dtype=float)
            for j in range(n):
                q += np.array([s.x3d(j), s.y3d(j), s.z3d(j)])
            q /= n
            centers.append(q)
        else:
            centers.append(np.zeros((3,), dtype=float))
    return np.array(centers)

def load_file(h, file_name):
    h('load_file("' + file_name + '")')

def setup_simulation(h, simulation_time_ms):
    h.tstop = simulation_time_ms
    if 'tvec' not in dir(h):
        h('objref tvec')
        h('tvec = new Vector()')
        h('tvec.record(&t)')


def add_IClamp_to_section(h, section, t_start, t_duration, amplitude_nA, location=0.5):
    iclamp = h.IClamp(section(location))
    iclamp.delay = t_start
    iclamp.amp = amplitude_nA
    iclamp.dur = t_duration
    return iclamp


def print_model_parameters(h, sections, section_names, channel_list):

    for i in range(len(sections)):
        print(section_names[i])
        print('\tL =', sections[i].L)
        print('\tdiam =', sections[i].diam)
        print('\tcm =', sections[i].cm)
        print('\tRa =', sections[i].Ra)
        try:
            print('\tek =', sections[i].ek)
        except:
            pass

        try:
            print('\tena =', sections[i].ena)
        except:
            pass

        try:
            print('\teca =', sections[i].eca)
        except:
            pass

        for s in sections[i]:
            print('\tsegment =', s)

            print('\t\tdistance =', h.distance(s))
            print('\t\tdiam =', s.diam)

            mechanisms = [s for s in dir(s) if s in channel_list]

            for m in mechanisms:
                print('\t\t' + m, end=' --- ')
                mech = getattr(s, m)
                parameters = [s for s in dir(mech) if '__' not in s]
                values = [getattr(mech, p) for p in parameters]

                for k in range(len(parameters)):
                    if type(values[k]) == float:
                        print(parameters[k] + ': ' + str(values[k]), end=', ')

                print('')

def setup_neuron(mechanism_library_filename):
    # Set up the hoc interpreter and load the mechanisms
    h = neuron.hoc.HocObject()
    h.load_file('stdrun.hoc')
    mechanism_library_filename = mechanism_library_filename
    try:
        h.nrn_load_dll(mechanism_library_filename)
    except:
        raise Exception(
            'Unable to load modules. Make sure to run nrnivmodl in directory with mod files and supply the correct link to the libnrnmech.dll or libnrnmech.so file.')
    return h
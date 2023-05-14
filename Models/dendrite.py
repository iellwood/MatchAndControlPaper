"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

A single tuft dendrite
"""


import HocPythonTools

class Dendrite:

    dendrite_number = -1


    def __init__(self, h, dendrite_parameters, name=None, nseg=None, diameter=None):

        Dendrite.dendrite_number += 1

        self.h = h

        if name is None:
            self.name = 'dendrite_' + str(Dendrite.dendrite_number)
        else:
            self.name = name

        self.section = h.Section(name='dendrite')
        self.section.L = dendrite_parameters['length']
        if nseg is None:
            self.section.nseg = 1 + 2 * int(dendrite_parameters['length'] / 2)
        else:
            self.section.nseg = nseg
        if diameter is None:
            self.section.diam = dendrite_parameters['dendrite diameter']
        else:
            self.section.diam = diameter
        self.section.cm = dendrite_parameters['cm']
        self.section.Ra = dendrite_parameters['Ra']

    def add_mechansim(self, mechanism_dict):
        HocPythonTools.add_mechanism(self.h, self.section, mechanism_dict)

    def add_mechanisms(self, mechanism_dict_list):
        for mechanism_dict in mechanism_dict_list:
            HocPythonTools.add_mechanism(self.h, self.section, mechanism_dict)

    def add_spine(self, spine, location):
        spine.connect(self.section(location))
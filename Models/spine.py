"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

A single spine
"""


import HocPythonTools

class Spine:
    spine_number = -1
    def __init__(self, h, spine_parameter_dict, name=None):
        Spine.spine_number += 1

        self.h = h

        if name is None:
            self.name = 'spine_' + str(Spine.spine_number) + '_'
        else:
            self.name = name + '_'

        self.head = h.Section(name=self.name + 'head')
        self.neck = h.Section(name=self.name + 'neck')

        self.head.L     = spine_parameter_dict['head_length']
        self.head.diam  = spine_parameter_dict['head_diameter']
        self.head.cm    = spine_parameter_dict['cm']
        self.head.Ra    = spine_parameter_dict['Ra']
        self.neck.L     = spine_parameter_dict['neck_length']
        self.neck.diam  = spine_parameter_dict['neck_diameter']
        self.neck.cm    = spine_parameter_dict['cm']
        self.neck.Ra    = spine_parameter_dict['Ra']

        self.head.connect(self.neck(1))

    def connect(self, section):
        self.neck.connect(section)

    def add_spine_mechanism(self, mechanism_properties, part=None):
        if part is None or part == 'spine' or part == 'all':
            HocPythonTools.add_mechanism(self.h, self.head, mechanism_properties)
            HocPythonTools.add_mechanism(self.h, self.neck, mechanism_properties)
        elif part == 'head':
            HocPythonTools.add_mechanism(self.h, self.head, mechanism_properties)
        elif part == 'neck':
            HocPythonTools.add_mechanism(self.h, self.neck, mechanism_properties)

    def add_spine_mechanisms(self, mechanism_properties_list, part=None):
        for mechanism_properties in mechanism_properties_list:
            self.add_spine_mechanism(self, mechanism_properties, part=part)
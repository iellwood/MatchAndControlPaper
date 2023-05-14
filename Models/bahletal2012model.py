"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

Reimplementation of Bahl Et. Al. 2012 in python
"""

import numpy as np
import HocPythonTools
import Models.basic_properties as basic_properties

class L5NeuronReducedModel_BahlEtAl2012:

    def __init__(self, h, include_apical=True, include_tuft=True):
        self.h = h

        self.include_apical = include_apical
        self.include_tuft = include_tuft

        self.iclamps = []
        self.soma = h.Section(name='soma')
        self.axon_hillock = h.Section(name='axon_hillock')
        self.axon_initial_segment = h.Section(name='axon_initial_segment')
        self.axon = h.Section(name='axon')
        self.basal = h.Section(name='basal')
        if include_apical:
            self.apical = h.Section(name='apical')
        if include_apical and include_tuft:
            self.tuft = h.Section(name='tuft')

        # Connect segments

        self.basal.connect(self.soma(0.5))
        if include_apical:
            self.apical.connect(self.soma(1.0))
        if include_apical and include_tuft:
            self.tuft.connect(self.apical(1))

        self.axon_hillock.connect(self.soma(0))
        self.axon_initial_segment.connect(self.axon_hillock(1))
        self.axon.connect(self.axon_initial_segment(1))

        # Initialize geometric and basic electrical parameters

        self.soma_area = 1682.96028429
        self.basal_area = 7060.90626796
        self.apicalshaftoblique_area = 9312.38528764
        self.tuft_area = 9434.24861189

        if include_apical:
            self.apical.nseg = 5
            self.apical.diam = self.apicalshaftoblique_area/np.pi/500
            self.apical.Ra = 261
            self.apical.L = 500

        if include_tuft and include_apical:
            self.tuft.nseg = 2
            self.tuft.L = 499
            self.tuft.Ra = 527
            self.tuft.diam = self.tuft_area/np.pi/499

        self.basal.nseg = 1
        self.basal.L = 257
        self.basal.Ra = 734
        self.basal.diam = self.basal_area/np.pi/257

        self.soma.diam = np.sqrt(self.soma_area/np.pi)
        self.soma.L = np.sqrt(self.soma_area/np.pi)
        self.soma.Ra = 82
        self.soma.nseg = 1
        self.h.distance(0, self.soma(0.0))

        self.axon.L = 500
        self.axon.diam = 1.5
        self.axon.Ra = 82
        self.axon.nseg = 1

        self.axon_initial_segment.L = 25
        self.axon_initial_segment.nseg = 5
        L5NeuronReducedModel_BahlEtAl2012.taper_diam(self.axon_initial_segment, 2.0, 1.5)
        self.axon_initial_segment.Ra = 82

        self.axon_hillock.L = 20
        self.axon_hillock.nseg = 5
        L5NeuronReducedModel_BahlEtAl2012.taper_diam(self.axon_hillock, 3.5, 2.0)
        self.axon_hillock.Ra = 82

        # Make lists for categories of segments

        self.axosomatic_list = [self.soma, self.basal, self.axon_hillock, self.axon_initial_segment, self.axon]
        self.axon_list = [self.axon_hillock, self.axon_initial_segment, self.axon]
        self.soma_list = [self.soma]
        self.basal_list = [self.basal]

        if include_apical:
            self.apicalshaftoblique_list = [self.apical]

        if include_apical and include_tuft:
            self.nat_list = [self.soma, self.axon_hillock, self.axon_initial_segment, self.apical, self.tuft]
            self.kfast_list = [self.soma, self.apical, self.tuft]
            self.kslow_list = [self.soma, self.apical, self.tuft]
            self.all_sections = [self.soma, self.basal, self.apical, self.tuft, self.axon, self.axon_initial_segment,
                                 self.axon_hillock]
            self.all_section_names = ['soma', 'basal', 'apical dendrite', 'tuft', 'axon', 'axon initial segment',
                                      'axon hillock']
            self.ih_list = [self.basal, self.apical, self.tuft]
            self.apicaltree_list = [self.apical, self.tuft]
            self.tuft_list = [self.tuft]



        elif include_apical:
            self.nat_list = [self.soma, self.axon_hillock, self.axon_initial_segment, self.apical]
            self.kfast_list = [self.soma, self.apical]
            self.kslow_list = [self.soma, self.apical]
            self.all_sections = [self.soma, self.basal, self.apical, self.axon, self.axon_initial_segment, self.axon_hillock]
            self.all_section_names = ['soma', 'basal', 'apical dendrite', 'axon', 'axon initial segment', 'axon hillock']
            self.ih_list = [self.basal, self.apical]
            self.apicaltree_list = [self.apical]


        else:
            self.nat_list = [self.soma, self.axon_hillock, self.axon_initial_segment]
            self.kfast_list = [self.soma]
            self.kslow_list = [self.soma]
            self.all_sections = [self.soma, self.basal, self.axon, self.axon_initial_segment, self.axon_hillock]
            self.all_section_names = ['soma', 'basal', 'axon', 'axon initial segment', 'axon hillock']
            self.ih_list = [self.basal]

        for section in self.all_sections:
            HocPythonTools.add_mechanism(h, section, basic_properties.BahlEtAl2012_pas)

        for section in self.ih_list:
            HocPythonTools.add_mechanism(h, section, basic_properties.BahlEtAl2012_ih)

        for section in self.nat_list:
            HocPythonTools.add_mechanism(h, section, basic_properties.BahlEtAl2012_nat)

        for section in self.kfast_list:
            HocPythonTools.add_mechanism(h, section, basic_properties.BahlEtAl2012_kfast)

        for section in self.kslow_list:
            HocPythonTools.add_mechanism(h, section, basic_properties.BahlEtAl2012_kslow)

        HocPythonTools.add_mechanism(h, self.soma, basic_properties.BahlEtAl2012_nap)
        HocPythonTools.add_mechanism(h, self.soma, basic_properties.BahlEtAl2012_km)
        if include_apical and include_tuft:
            HocPythonTools.add_mechanism(h, self.tuft, basic_properties.BahlEtAl2012_cad)
            HocPythonTools.add_mechanism(h, self.tuft, basic_properties.BahlEtAl2012_sca)
            HocPythonTools.add_mechanism(h, self.tuft, basic_properties.BahlEtAl2012_kca)

        self.Rm_axosomatic = 15000
        self.spinefactor = 2.0
        self.decay_kfast = 50.0
        self.decay_kslow = 50.0

        if self.include_apical and self.include_tuft:
            self.h.ion_style("ca_ion", 0, 1, 0, 0, 0, sec=self.tuft)

        self.set_parameters_from_BahlEtAl_2012_init_model1_hoc_file()
        self.recalculate_passive_properties()
        self.recalculate_channel_densities()

        # add recordings
        self.t = h.Vector().record(h._ref_t)

        self.voltage_recordings = [self.h.Vector().record(section(0.5)._ref_v) for section in self.all_sections]

        if include_apical:
            self.apical_voltage_recordings = [self.h.Vector().record(s._ref_v) for s in self.apical]

        self.h.celsius = 37

        #self.make_modifications_for_match_and_control()




    def recalculate_passive_properties(self):

        for section in self.axosomatic_list:
            for seg in section: seg.pas.g = 1.0/self.Rm_axosomatic

        if self.include_apical:
            for section in self.apicaltree_list:
                for seg in section: seg.pas.g = self.soma(0.5).pas.g * self.spinefactor
                for seg in section: seg.cm = self.soma.cm * self.spinefactor

    def recalculate_channel_densities(self):
        if self.include_apical:
            for section in self.apicaltree_list:
                for seg in section:
                    seg.BahlEtAl_2012_kfast.gbar = self.soma(0.5).BahlEtAl_2012_kfast.gbar * np.exp(-self.h.distance(seg)/self.decay_kfast)
                    seg.BahlEtAl_2012_kslow.gbar = self.soma(0.5).BahlEtAl_2012_kslow.gbar * np.exp(-self.h.distance(seg)/self.decay_kslow)
        if self.include_apical and self.include_tuft:
            tuft_segs = [s for s in self.tuft]
            mih = tuft_segs[0].BahlEtAl_2012_ih.gbar/self.h.distance(self.tuft(0))
        if self.include_apical:
            mnat = -0.5313787051835502 # Attempt to reproduce code that computes this number failed, so hard coding it

            for seg in self.apical:
                seg.BahlEtAl_2012_nat.gbar = mnat * self.h.distance(seg) + self.soma(0.5).gbar_BahlEtAl_2012_nat
                seg.BahlEtAl_2012_ih.gbar = mih * self.h.distance(seg)

            # Unknown why the formula for the very last segment fails to reproduce the right conductances from BahlEtAl2012
            apical_segments = [seg for seg in self.apical]
            apical_segments[-1].BahlEtAl_2012_nat.gbar = 6.558244000000002
            apical_segments[-1].BahlEtAl_2012_ih.gbar = 17.694744
            apical_segments[-1].BahlEtAl_2012_kfast.gbar = 0.006660875706745579
            apical_segments[-1].BahlEtAl_2012_kslow.gbar = 0.0014975152688699454

        if self.include_tuft and self.include_apical:
            tuft_segs[-1].BahlEtAl_2012_kfast.gbar = 1.3193667752182392e-06
            tuft_segs[-1].BahlEtAl_2012_kslow.gbar = 1.0989351886184045e-08



    def set_parameters_from_BahlEtAl_2012_init_model1_hoc_file(self):

        for section in self.all_sections:
            for seg in section:
                seg.pas.e = -83.056442

        self.Rm_axosomatic = 23823.061083

        for section in self.axosomatic_list:
            section.cm = 2.298892

        self.spinefactor = 0.860211

        for seg in self.soma: seg.BahlEtAl_2012_nat.gbar = 284.546493
        for seg in self.soma: seg.BahlEtAl_2012_kfast.gbar = 50.802287
        for seg in self.soma: seg.BahlEtAl_2012_kslow.gbar = 361.584735
        for seg in self.soma: seg.BahlEtAl_2012_nap.gbar = 0.873246

        for seg in self.soma: seg.BahlEtAl_2012_km.gbar = 7.123963

        for seg in self.basal: seg.BahlEtAl_2012_ih.gbar = 15.709707
        if self.include_tuft and self.include_apical:
            for seg in self.tuft: seg.BahlEtAl_2012_ih.gbar = 17.694744
            for seg in self.tuft: seg.BahlEtAl_2012_nat.gbar = 6.558244

        self.decay_kfast = 58.520995
        self.decay_kslow = 42.208044

        for seg in self.axon_hillock: seg.BahlEtAl_2012_nat.gbar = 8810.657100

        for seg in self.axon_initial_segment: seg.BahlEtAl_2012_nat.gbar = 13490.395442
        for seg in self.axon_initial_segment: seg.BahlEtAl_2012_nat.vshift = 10
        for seg in self.axon_initial_segment: seg.BahlEtAl_2012_nat.vshift2 = -9.802976

        if self.include_apical:
            Ra_apical = 454.05939784
            self.apical.Ra = Ra_apical

        if self.include_apical and self.include_tuft:
            for seg in self.tuft: seg.BahlEtAl_2012_sca.gbar = 3.67649485
            for seg in self.tuft: seg.BahlEtAl_2012_sca.vshift = 7.4783781
            for seg in self.tuft: seg.BahlEtAl_2012_kca.gbar = 9.75672674


    @staticmethod
    def taper_diam(sec, zero_bound, one_bound):
        dx = 1. / sec.nseg
        diameters = np.arange(dx / 2, 1, dx)
        for i, seg in enumerate(sec):
            seg.diam = (one_bound - zero_bound) * diameters[i] + zero_bound

    def add_iclamp_to_soma(self, start_time, duration, current):
        iclamp = HocPythonTools.add_IClamp_to_section(self.h, self.soma, start_time, duration, current)
        self.iclamps.append(iclamp)


    def record(self, duration_ms):

        self.h.finitialize(-65)
        self.h.continuerun(duration_ms)

        self.iclamps = []

        return np.array(self.t), np.array(self.voltage_recordings)

    def print_parameters(self):
        channels = ['pas', 'BahlEtAl_2012_cad', 'BahlEtAl_2012_ih', 'BahlEtAl_2012_km', 'BahlEtAl_2012_kca', 'BahlEtAl_2012_kfast', 'BahlEtAl_2012_kslow', 'BahlEtAl_2012_nap', 'BahlEtAl_2012_nat', 'BahlEtAl_2012_sca']

        HocPythonTools.print_model_parameters(self.h, self.all_sections, self.all_section_names, channels)


    def add_iclamps(self, stimulation_times_ms, stimulation_duration_ms, amplitude_nA):
        self.iclamps = []
        for i in range(len(stimulation_times_ms)):
            stimobj = self.h.IClamp(self.soma(0.5))
            stimobj.delay = stimulation_times_ms[i]
            stimobj.amp = amplitude_nA
            stimobj.dur = stimulation_duration_ms
            self.iclamps.append(stimobj)

    def make_modifications_for_match_and_control(self):
        self.apical.Ra = 100
        self.apical.cm = 1
        self.apical.nseg = 100

        distances = [-100, 7.689268394754731, 23.067805184264195, 38.44634197377366, 56.985559682117945, 78.68545830929705,
         100.38535693647617, 125.67107819383484, 146.98787721015685, 160.74993135526256, 174.51198550036827,
         185.11096666463686, 199.59568829830448, 221.12922338220824, 242.662758466112, 261.0883219287762,
         276.4059137702009, 291.7235056116256, 306.9881950262975, 325.7488131054729, 348.05826227590484,
         370.3677114463368, 390.95570979101336, 409.8222573099346, 428.6888048288559, 447.55535234777716,
         466.4218998666984, 485.2884473856197, 504.154994904541, 523.0215424234623, 541.8880899423835,
         560.7546374613048, 579.6211849802261, 598.4877324991473, 617.3542800180687, 1000000]
        diams = [7.20486823863129, 7.20486823863129, 4.08698291062178, 3.6502690127734483, 3.2584230635936025, 3.089999914169311,
                 3.089999914169311, 3.089999914169311, 3.089999914169311, 3.089999914169311, 3.08999991416931,
                 3.089999914169311, 2.569999933242798, 2.569999933242798, 2.5962909540493606, 2.5699999332427974,
                 2.451774938921602, 2.6864140514981663, 3.089999914169312, 3.055301743259446, 2.494606981761777,
                 2.5699999332427974, 2.31302307025813, 2.5699999332427983, 2.569999933242797, 2.5699999332427987,
                 2.569999933242797, 2.569999933242797, 2.529957657700846, 2.4821465756433225, 2.8299999237060534,
                 2.8299999237060534, 2.8299999237060534, 2.829999923706058, 2.8299999237060534, 2.8299999999999]

        distances = np.array(distances)
        diams = np.array(diams)

        hayetal_diameters = np.interp([self.h.distance(s) - 20 for s in self.apical], distances, diams)
        for i, seg in enumerate(self.apical):
            seg.diam = hayetal_diameters[i]
            print('seg.diam =', seg.diam, 'distance =', self.h.distance(seg) - 20)
        # self.apical.diam = 0.5

        for seg in self.apical:
            if seg.BahlEtAl_2012_nat.gbar < 100:
                seg.BahlEtAl_2012_nat.gbar = 100


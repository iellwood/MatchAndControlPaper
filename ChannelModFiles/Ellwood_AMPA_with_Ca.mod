TITLE AMPA Synaptic current

COMMENT

This is AMPA model is based code from Kim et. al 2015, Badoual 2016 and Humphries et. al 2021 The main addition is a parameter that keeps track of the
integral of the fourth power of the calcium concentration. When this calcium concentration crosses a threshold, the AMPA
channel is potentiated.

ENDCOMMENT

: Declare public variables


NEURON {
	POINT_PROCESS AMPA_WITH_CA
	USEION ca READ cai
	RANGE  e, i, tau1, tau2, g, mg, gmax, tau_plasticity, ca_potentiation_threshold, ca_zero_point, sigmoid_sharpness, max_potentiation, tau_g_dynamic_delayed, sigmoid_offset
	NONSPECIFIC_CURRENT i, iampa
	GLOBAL total
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
	(mM) = (milli/liter)
}

PARAMETER {
	tau1 	= 0.34 		(ms)			  : tau of alpha synapse for AMPA-R
	tau2 	= 2.0 		(ms)
	e 		= 0.0		(mV)
	mg 		= 1 		(mM) 			  : external magnesium concentration
	sh 		= 0 		(mV)
	gmax 	= 0.002 	(uS)

	ca_zero_point = 500 (mM)
	c_ca_4  = 0.13
	c_ca_2  = 0.01
	lambda_1 = 5
	normalization = 7
	max_potentiation = 8

	ca_potentiation_threshold = 1e50
	sigmoid_sharpness = 20
	sigmoid_offset = 0

	: Parameters needed for calcium-dependent plasticity

	cai		(mM)		                  : Ca concentration inside
	tau_plasticity     = 20000.0
	tau_g_dynamic_delayed = 500
	alpha = 0
}

ASSIGNED {
	v (mV)
	i (nA)
	g (uS)
	iampa (nA)
	factor
	total (uS)
}

STATE {

	Aampa (uS)
	Gampa (uS)
	g_dynamic ()
	g_dynamic_delayed ()
}

INITIAL {
	LOCAL tp
	total = 0
	if (tau1/tau2 > .9999) {
		tau1 = .9999*tau2
	}
	Aampa = 0
	Gampa = 0
	g_dynamic = 0
	g_dynamic_delayed = 0


	tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau2)
	factor = 1/factor

}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = Gampa - Aampa
	alpha = sig(sigmoid_sharpness*(g_dynamic_delayed/ca_potentiation_threshold - 1) + sigmoid_offset)
	iampa = ((1 - alpha) + max_potentiation * alpha) * gmax * g * (v - e)

	i = iampa
}

DERIVATIVE state {
	Aampa' = - Aampa/tau1
	Gampa' = - Gampa/tau2 : - Gampa/tau1	Aampa -> Gampa -> disappear with rate const of 1/tau1.

    g_dynamic' = -(g_dynamic)/tau_plasticity + (cai - 5e-7) * (cai - 5e-7) * (cai - 5e-7) * (cai - 5e-7)
    g_dynamic_delayed' = (g_dynamic - g_dynamic_delayed)/tau_g_dynamic_delayed
}

NET_RECEIVE(weight (uS)) {
	state_discontinuity(Aampa, Aampa + weight*factor)
	state_discontinuity(Gampa, Gampa + weight*factor)
	total = total+weight
}

FUNCTION norm(x){
    if(x >= 7){
        norm = 7
    } else {
        norm = x
    }
}

FUNCTION sig(x){
    if(x > 0) {
        sig = 1/(1 + exp(-x))
    } else {
        sig = exp(x)/(exp(x) + 1)
    }
}
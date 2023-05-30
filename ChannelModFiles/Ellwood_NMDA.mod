TITLE dual-exponential model of NMDA receptors

COMMENT

This is NMDA model is an amalgam of code from Kim et. al 2015, Badoual 2016 and Humphries et. al. 2021, which
are themselves based on older sources.

ENDCOMMENT

NEURON {
	POINT_PROCESS NMDA
	NONSPECIFIC_CURRENT i
	USEION ca READ cai WRITE ica
	RANGE g, gmax, tau1, tau2, e, i, wf
	THREADSAFE
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
	(mM) = (milli/liter)
	(S)  = (siemens)
	(pS) = (picosiemens)
	(um) = (micron)
	(J)  = (joules)
	(celsius) = (degC)
}

PARAMETER {
    gmax = 0.05     (uS)
	tau1 = 2.0      (ms)
	tau2 = 26.0     (ms)
	celsius = 37	(degC)
	e = -0.7		(mV)
	cao = 1.5	        (mM)		        : Ca concentration outside the cell
	cai		            (mM)		        : Ca concentration inside
    g
}


ASSIGNED {
	ica     (nA)        : calcium current
	v		(mV)
	dt		(ms)
	i		(nA)
	factor
	wf
}

STATE {
	A
	B
}

INITIAL {
	LOCAL tp
	if (tau1/tau2 > .9999) {
		tau1 = .9999*tau2
	}
	A = 0
	B = 0
	tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau2)
	factor = 1/factor

	wf = 1
	mgblock(v)
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = gmax*(B - A)*mgblock(v)
	ica = (0.021) * g * (v - 130) : 10% of the current is calcium at -40 mV
	i = g*(v - e) - ica
}

DERIVATIVE state {
	A' = -A/tau1
	B' = -B/tau2
}

NET_RECEIVE(weight) {
	wf = weight*factor
	A = A + wf
	B = B + wf
}

FUNCTION mgblock(v(mV)) {
        TABLE
        FROM -140 TO 80 WITH 1000

        : Following Kim et. al. 2015, I have shifted the half-open voltage to -10 mV
        : There seems to be some disagreement about the precise voltage dependence of the magnesium block
        : and the simple sigmoid used in these models is likely a simple approximation.
        : Note that the factor of 0.075 also varies across the literature and Kim et. al. tried different values.
        : It may be useful to test the model's performance on this parameter in the future.

        mgblock = 1 / (1 + exp(-0.075 (/mV) * (v - (-10))))
}

FUNCTION exptable(x) {
        TABLE  FROM -10 TO 10 WITH 2000

        if ((x > -10) && (x < 10)) {
                exptable = exp(x)
        } else {
                exptable = 0.
        }
}
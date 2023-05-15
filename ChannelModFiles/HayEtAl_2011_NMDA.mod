TITLE dual-exponential model of NMDA receptors

COMMENT
Classic double-exponential model of NMDAR
Mg++ voltage dependency from Spruston95 -> Woodhull, 1973
Keivan Moradi 2011

--- (and now back to the original exp2syn comments) ---

Two state kinetic scheme synapse described by rise time tau1,
and decay time constant tau2. The normalized peak condunductance is 1.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tau1 and 1/tau2 is
 A = a*exp(-t/tau1) and
 G = a*tau2/(tau2-tau1)*(-exp(-t/tau1) + exp(-t/tau2))
	where tau1 < tau2

If tau2-tau1 -> 0 then we have a alphasynapse.
and if tau1 -> 0 then we have just single exponential decay.

The factor is evaluated in the
initial block such that an event of weight 1 generates a
peak conductance of 1.

In the initial block we initialize the factor and total and A and B to starting values. The factor is
defined in terms of tp, a local variable which defined the time of the peak of the function as
determined by the tau1 and tau2.  tp is the maximum of the function exp(-t/tau2) – exp(-t/tau1).  To
verify this for yourself, take the derivative, set it to 0 and solve for t.  The result is tp as defined
here. Factor is the value of this function at time tp, and 1/factor is the normalization applied so
that the peak is 1.  Then the synaptic weight determines the maximum synaptic conductance.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

ENDCOMMENT

NEURON {
	POINT_PROCESS NMDA
	NONSPECIFIC_CURRENT i
	USEION ca READ cai WRITE ica

	RANGE g, gmax, tau1, tau2, e, i, mg, K0, delta, wf
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
	mg = 1			(mM)
	K0 = 4.1		(mM)	: IC50 at 0 mV from Spruston95
	delta = 0.8 	(1)		: the electrical distance of the Mg2+ binding site from the outside of the membrane from Spruston95
: Parameter Controls Ohm's law in NMDAR
	e = -0.7		(mV)	: in CA1-CA3 region = -0.7 from Spruston95

: Parameters from nmdacin (a.k.a., NMDAKIT)
	Px=4.6925	(cm3 mV/coulomb)	:determined empirically such as 10% of the current is Ca2+ current at -40mV
	temp_NMDA = 37		(degC)
	cao = 1.5	(mM)		        : Ca concentration outside the cell
	cai		(mM)		            : Ca concentration inside
    g
}

CONSTANT {
	T = 273.16	(degC)
	F = 9.648e4	(coul)	: Faraday's constant (coulombs/mol)
	R = 8.315	(J/degC): universal gas constant (joules/mol/K)
	z = 2		(1)		: valency of Mg2+
}

ASSIGNED {
	ica     (mA/cm2)        : calcium current
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
   	ica = (0.001) * g * (0.051(cm3/coulomb)*v+Px)   *   (4.0*v*F*F / (R * (temp_NMDA+273) ))   *   (-cao*exptable(-2*v*F/(R* (temp_NMDA+273) ))  +  cai)  /  (1.0 - exptable(-2.0*v*F/(R* (temp_NMDA+273) )))
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
        DEPEND mg
        FROM -140 TO 80 WITH 1000
        : from Jahr & Stevens
        : ORIGINAL: mgblock = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))

        : Following Kim et. al. 2015, I have shifted the half-open voltage to -10 mV
        : There seems to be some disagreement about the precise voltage dependence of the magnesium block
        : and the simple sigmoid used in these models is likely a simple approximation.
        : Without this shift, the NMDA receptors readily open even at -75 mV.
        : Note that the factor of 0.075 also varies across the literature.
        : We are using (0.001) * (-z) * delta * F/R/(T + celcius)

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
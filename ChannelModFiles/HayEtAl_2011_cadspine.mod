TITLE decay of internal calcium concentration
:
: Internal calcium concentration due to calcium currents and pump.
: Differential equations.
:
: Simple model of ATPase pump with 3 kinetic constants (Destexhe 92)
:     Cai + P <-> CaP -> Cao + P  (k1,k2,k3)
: A Michaelis-Menten approximation is assumed, which reduces the complexity
: of the system to 2 parameters:
:       kt = <tot enzyme concentration> * k3  -> TIME CONSTANT OF THE PUMP
:	kd = k2/k1 (dissociation constant)    -> EQUILIBRIUM CALCIUM VALUE
: The values of these parameters are chosen assuming a high affinity of
: the pump to calcium and a low transport capacity (cfr. Blaustein,
: TINS, 11: 438, 1988, and references therein).
:
: Units checked using "modlunit" -> factor 10000 needed in ca entry
:
: VERSION OF PUMP + DECAY (decay can be viewed as simplified buffering)
:
: All variables are range variables
:
:
: This mechanism was published in:  Destexhe, A. Babloyantz, A. and
: Sejnowski, TJ.  Ionic mechanisms for intrinsic slow oscillations in
: thalamic relay neurons. Biophys. J. 65: 1538-1552, 1993)
:
: Written by Alain Destexhe, Salk Institute, Nov 12, 1992
:modified by M. Badoual, 2004

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX cadspine
	USEION ca READ ica, cai WRITE cai
	RANGE ca
	GLOBAL depth,cainf,taur,cadend,D,lneck,sneck,vspine
}

UNITS {
	(molar) = (1/liter)	    : moles do not appear in units
	(mM)	= (millimolar)
	(um)	= (micron)
	(mA)	= (milliamp)
	(msM)	= (ms mM)
	FARADAY = (faraday) (coulomb)
}


PARAMETER {
	depth	    = 0.125	        (um)        : depth of shell
	taur	    = 15	        (ms)        : rate of calcium removal
	cainf	    = 5e-7          (mM)
	cai		                    (mM)
	D           = 0.22	        (um2/ms)    : diffusion coefficient of the calcium
	cadend      = 1e-4          (mM)        : calcium concentration in dendrite
	lneck       = 1.0	        (um)        : length of the neck
	sneck       = 0.007853981   (um2)       : cross section of the neck
	vspine      = 0.098174770   (um3)       : volume of the spine
}

STATE {
	ca		(mM)
}

INITIAL {
	ca = cainf
	cai=ca
}

ASSIGNED {
	ica		(mA/cm2)
	drive_channel	(mM/ms)

}

BREAKPOINT {
	SOLVE state METHOD derivimplicit : see http://www.neuron.yale.edu/phpBB/viewtopic.php?f=28&t=592
}

DERIVATIVE state {

	drive_channel =  - (10000) * ica / (2 * FARADAY * depth * 18.0)
	if (drive_channel <= 0.0) { drive_channel = 0.0 }	: cannot pump inward

	ca' = drive_channel + (cainf-ca)/taur : - D*(ca-cadend)*sneck/(lneck*vspine)

	cai = ca
}

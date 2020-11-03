# -*- coding: utf-8 -*-
from snewpy.FlavorTransformation import *

from astropy import units as u
from numpy import sin, cos

def test_mixing_angles():
    assert(theta12 == 33.44 * u.degree)
    assert(theta13 ==  8.57 * u.degree)
    assert(theta23 == 49.0 * u.degree)

def test_noxform():
    # No transformation.
    xform = NoTransformation()
    assert(xform.p()==1 and xform.pbar()==1)

def test_adiabaticmsw_nmo():
    # Adiabatic MSW (normal ordering).
    xform = AdiabaticMSW_NMO()
    assert(xform.p()==sin(theta13)**2)
    assert(xform.pbar()==(cos(theta12)*cos(theta13))**2)

def test_adiabaticmsw_imo():
    # Adiabatic MSW (inverted ordering).
    xform = AdiabaticMSW_IMO()
    assert(xform.p()==(sin(theta12)*cos(theta13))**2)
    assert(xform.pbar()==sin(theta13)**2)

def test_2fd():
    # Two-flavor decoherence.
    xform = TwoFlavorDecoherence()
    assert(xform.p()==0.5)
    assert(xform.pbar()==0.5)

def test_3fd():
    # Three-flavor decoherence.
    xform = ThreeFlavorDecoherence()
    assert(xform.p()==1./3)
    assert(xform.pbar()==1./3)

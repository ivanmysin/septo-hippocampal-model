# -*- coding: utf-8 -*-
"""
main script
"""

import lib
import numpy as np
import matplotlib.pyplot as plt

def getFullVhisroty(obj):
    N = obj.getNVhistvalues()
    Vhist = np.empty(N, dtype=np.float64)
    for idx in range(N):
        Vhist[idx] = obj.getVhistbyIdx(idx)
    return Vhist


soma_params = {
    "V0": 0.0,
    "C" : 3.0,
    "Iextmean": 4.5,        
    "Iextvarience": 0.9,
    "ENa": 120.0,
    "EK": -15.0,
    "El": 0.0,
    "ECa": 140.0,
    "CCa": 0.05,
    "sfica": 0.13,
    "sbetaca": 0.075, 
    "gbarNa": 30.0,
    "gbarK_DR": 17.0,
    "gbarK_AHP": 0.8,        
    "gbarK_C " : 15.0,
    "gl": 0.1,
    "gbarCa": 6.0,
}

comp = lib.pyCompartment(soma_params)
comp.integrate(0.1, 1000)
V = comp.getVhist()
plt.plot(V)
print (comp.test())

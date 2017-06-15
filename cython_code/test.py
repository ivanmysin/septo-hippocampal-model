#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test
"""

import lib2 as lib
import matplotlib.pyplot as plt

HS_neuron_params = {
        "V0" : -63.0,
        "Iextmean" : -0.5,        
        "Iextvarience" : 0.0,
        "ENa" : 55.0,
        "EK"  : -90.0,
        "El"  : -65.0,
        "EH"  : -40.0,
        "ECa" : 120.0,
        
        "Ca"  : 0.0,
        "KD"  : 30.0,
         
        "gbarNa" : 35.0,
        "gbarK"  : 9.0,
        "gl"     : 0.1,   
        "gbarK_Ca" : 10,
        "gbarCa" : 1.0,
        "gbarH" : 0.15,    
        "alpha_Ca" : 0.002,
        "tau_Ca"   : 80,
}



Neuron = lib.HS_projective_neuron(HS_neuron_params)
Neuron.integrate(0.05, 1000)

Vhist = Neuron.getVhist()
plt.plot(Vhist)
plt.show()
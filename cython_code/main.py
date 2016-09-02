# -*- coding: utf-8 -*-
"""
main script
"""

import lib
import numpy as np
import matplotlib.pyplot as plt


soma_params = {
    "name" : "soma",
    "params" : {
        "V0": 0.0,
        "C" : 3.0,
        "Iextmean": 0.0,        
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
}


dendrit_params = {
    "name": "dendrite",
    "params": {
        "V0": 0.0,
        "C" : 3.0,
        "Iextmean": 1.25,        
        "Iextvarience": 0.9,
        "ENa": 120.0,
        "EK": -15.0,
        "El": 0.0,
        "ECa": 140.0,
        "CCa": 0.05,
        "sfica": 0.13, 
        "sbetaca": 0.075, ######## ??????????
        "gbarNa": 0.0,
        "gbarK_DR": 0.0,
        "gbarK_AHP": 0.8,        
        "gbarK_C " : 5.0,
        "gl": 0.1,
        "gbarCa": 5.0,
    }
}

connection_params = {
    "name": "soma-dendrite", 
    "compartment1": "soma",
    "compartment2": "dendrite",
    "p": 0.5,
    "g": 1.5,
}

neuron = lib.pyComplexNeuron([soma_params, dendrit_params], [connection_params])
neuron.integrate(0.1, 1000)
V = neuron.getVhistByCompartmentName("soma")
plt.plot(V)


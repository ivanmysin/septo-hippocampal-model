# -*- coding: utf-8 -*-
"""
main script
"""

import lib2 as lib
import numpy as np
import matplotlib.pyplot as plt


soma_params = {
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


dendrite_params = {
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

connection_params = {
    "compartment1": "soma",
    "compartment2": "dendrite",
    "p": 0.5,
    "g": 1.5,
}

ext_synapce_params = {
    "Erev" : 60.0,
    "gbarS": 0.005,
    "tau" : 2.0,
    "w" : 1.0,
}
"""
neuron = lib.pyComplexNeuron([soma_params, dendrit_params], [connection_params])
neuron.integrate(0.1, 1000)
V = neuron.getVhistByCompartmentName("soma")
plt.plot(V)
"""
neurons = []
synapses = []
Nn = 20
Ns = 30

for idx in range(Nn):
    soma = {"soma" : soma_params.copy()}
    dendrite = {"dendrite" : dendrite_params.copy()}
    connection = connection_params.copy()
    neuron = {
        "compartments" : [soma, dendrite],
        "connections" : [connection]
    }
    neurons.append(neuron)
    
for idx in range(Ns):
    synapse = {
       "pre_ind": 0, 
       "post_ind": 1,
       "pre_compartment_name": "soma",
       "post_compartment_name" : "soma",
       "params": ext_synapce_params.copy()
    }
  
net = lib.Network(neurons, synapses) #
net.integrate(0.1, 1000)
#plt.plot(com.getVhist())
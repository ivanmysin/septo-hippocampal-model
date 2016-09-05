# -*- coding: utf-8 -*-
"""
main script
"""

import lib2 as lib
import numpy as np
import matplotlib.pyplot as plt

def iext_function(neuron_ind, compartment_name, t, params=None):
    Iext = 0
    
    if not(type(params) is None):
        iext_function.phase_shift = params
        
    if compartment_name == "soma":
        Iext = 2*np.cos(2*np.pi*t*8) - 2
    if compartment_name == "dendrite":
        Iext = 2*np.cos(2*np.pi*t*8 + iext_function.phase_shift) - 2
    
    return Iext

def run_model():
    soma_params = {
            "V0": 0.0,
            "C" : 3.0,
            "Iextmean": -1.5,        
            "Iextvarience": 1.0,
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
            "Iextmean": -1.5,        
            "Iextvarience": 1.0,
            "ENa": 120.0,
            "EK": -15.0,
            "El": 0.0,
            "ECa": 140.0,
            "CCa": 0.05,
            "sfica": 0.13, 
            "sbetaca": 0.075, 
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
    
    ext_synapse_params = {
        "Erev" : 60.0,
        "gbarS": 0.005,
        "tau" : 2.0,
        "w" : 10.0,
    }
    ###########################
    neurons = []
    synapses = []
    Nn = 100
    Ns = 300
    
    for idx in range(Nn):
        soma = {"soma" : soma_params.copy()}
        soma["soma"]["V0"] = 5 * np.random.randn()
        soma["soma"]["Iextmean"] += 0.1*np.random.randn()
        
        dendrite = {"dendrite" : dendrite_params.copy()}
        dendrite["dendrite"]["V0"] = soma["soma"]["V0"]
        dendrite["dendrite"]["Iextmean"] += 0.1*np.random.randn()
       
        connection = connection_params.copy()
        neuron = {
            "compartments" : [soma, dendrite],
            "connections" : [connection]
        }
        neurons.append(neuron)
        
    for idx in range(Ns):
        pre_ind = np.random.randint(0, Nn)
        post_ind = np.random.randint(0, Nn)
        if (pre_ind == post_ind):
            post_ind = np.random.randint(0, Nn)
        synapse = {
           "pre_ind": pre_ind, 
           "post_ind": post_ind,
           "pre_compartment_name": "soma",
           "post_compartment_name" : "soma",
           "params": ext_synapse_params.copy()
        }
        synapses.append(synapse)
      
    dt = 0.1
    duration = 1000
    net = lib.Network(neurons, synapses)
    net.integrate(dt, duration, iext_function)
    V = net.getVhist()
    VmeanSoma = 0
    VmeanDendrite = 0
    for v in V:
        VmeanSoma += v["soma"]
        VmeanDendrite += v["dendrite"]
    VmeanSoma /= Nn
    VmeanDendrite /= Nn
    t = np.linspace(0, duration, VmeanSoma.size)
    plt.figure()
    plt.subplot(211)
    plt.plot(t, VmeanSoma, "b")
    plt.subplot(212)
    plt.plot(t, VmeanDendrite, "r")
    
    plt.figure()
    Vsomahalf = VmeanSoma[t > duration/2]
    fft_y = np.abs(np.fft.rfft(Vsomahalf))/Vsomahalf.size
    fft_x = np.fft.rfftfreq(Vsomahalf.size, 0.001*dt)
    plt.plot(fft_x[1:], fft_y[1:])
    plt.xlim(2, 50)
    
    theta_power = np.sum(fft_y[(fft_x>4)&(fft_x<12)])/np.sum(fft_y)
    
    firing = net.getFiring()
    plt.figure()
    plt.plot(firing[0, :], firing[1, :], '.')
    plt.show()
    print (theta_power)
    return theta_power
saving_fig_path = "/home/ivan/Data/modeling_septo_hippocampal_model/"
theta_power = np.array([], dtype=float)
p = np.linspace(-np.pi, np.pi, 20)
for idx in range(100):
    if (idx%5 == 0):
        iext_function(0, 0, 0, params=p[idx])
    theta = run_model()
    theta_power = np.append(theta_power, theta)

plt.figure()
plt.plot(theta_power)
plt.savefig(saving_fig_path + "phase_shift.png")





    
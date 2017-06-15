#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
balancing basket to pyramide and CS to pyramide
"""

import lib2 as lib
import numpy as np
import os
import time



def run_model(path, params):
    soma_params = {
            "V0": 0.0,
            "C" : 3.0,
            "Iextmean": -1.5,        
            "Iextvarience": 0.5,
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
            "Iextvarience": 0.5,
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
    
    basket_fs_neuron = {
         "V0": -65.0,
         "Iextmean": -0.5,        
         "Iextvarience": 0.2,
         "ENa": 50.0,
         "EK": -90.0,
         "El": -65.0,
         "gbarNa": 55.0,
         "gbarK": 8.0,
         "gl": 0.1,   
         "fi": 10,
    }
    
    olm_params = {
        "V0" : -70,
        "gl" : 0.05,
        "El" : -70,
        "gbarNa" : 10.7, 
        "ENa" : 90,
        "gbarK" : 31.9,
        "EK" : -100,
        "gbarKa" : 16.5,
        "gbarH" : 0.05,
        "EH" : -32.9,
        "Iextmean" : 0.0,        
        "Iextvarience": 0.2,
    }
    
    CosSpikeGeneratorParams = {
       "freq"  : 5.0, # frequency in Hz
       "phase" : 0.0, # phase in rad
       "latency" :  10.0, # in ms
       "probability" : 0.01,
       "threshold" : 0.3,
    }
    
    PoisonSpikeGenerator = {
       "latency" :  10.0, # in ms
       "probability" : 0.00001,
    }
    
    simple_ext_synapse_params = {
        "Erev" : 60.0,
        "gbarS": 0.005,
        "tau" : 0.2,
        "w" : 20.0,
        "delay" : 0,
    }
    
    simple_inh_synapse_params = {
        "Erev" : -15.0,
        "gbarS": 0.005,
        "tau" : 0.5,
        "w" : 100.0,
        "delay" : 0,
    }
    
    ext_synapse_params = {
        "Erev" : 60.0,
        "gbarS": 0.005,
        "alpha_s" : 1.1,
        "beta_s": 0.19,
        "K": 5.0,
        "teta": 2.0,
        "w" : 20.0,
        "delay" : 0,
    }
    
    inh_synapse_params = {
        "Erev" : -15.0,
        "gbarS": 0.005,
        "alpha_s" : 14.0,
        "beta_s": 0.07,
        "K": 2.0,
        "teta": 0.0,
        "w" : 20.0,
        "delay" : 0,
    }
    
    
  
   
      
    neurons = []
    synapses = []
    Np = 400  # number of pyramide neurons
    Nb = 50   # number of basket cells
    Nolm = 50 # 50 # number of olm cells
    
    Npv1 = params["n_pv1"]   # number of septum spike generators    
    Npv2 = params["n_pv2"]
    
    Nec = params["n_ec"] # number of EC inputs
    Ncs = params["n_cs"] # 100 # number of Shaffer collateral iunput
    
    Ns = 600           # number synapses between pyramide cells
    Nbasket2pyr = params["Nsbas2pyr"] # 100 4 # number synapses from basket interneuron to one pyramide neuron 
    Nolm2pyr = params["Nsolm2pyr"]    # 4 # number synapses from basket interneuron to one pyramide neuron 
    Nspyr2basket = 10  # number synapses from one pyramide to interneurons
    Nspyr2OLM = 20    
   
    # number synapses from septal generators to hippocampal interneurons 
    Nspv12bas =  params["Nspv12bas"]
    Nspv22olm = params["Nspv22olm"]
    NsEc2pyr = params["NsEc2pyr"]     
    NsCs2pyr = params["NsCs2pyr"] 
    NsCs2bas = params["NsCs2bas"]
    
    indexes = {
        "pyr" : [],
        "bas" : [],
        "olm" : [],
        "pv1" : [],
        "pv2" : [],
        "mec" : [],
        "cs"  : [],
    }
            
    
    for idx in range(Np):
        soma = {"soma" : soma_params.copy()}
        soma["soma"]["V0"] = 0.5 * np.random.randn()
        soma["soma"]["Iextmean"] += 0.5 * np.random.randn()
        
        dendrite = {"dendrite" : dendrite_params.copy()}
        dendrite["dendrite"]["V0"] = soma["soma"]["V0"]
        dendrite["dendrite"]["Iextmean"] += 0.5 * np.random.randn()
       
        connection = connection_params.copy()
        neuron = {
            "type" : "pyramide",
            "compartments" : [soma, dendrite],
            "connections" : [connection]
        }
        indexes["pyr"].append(len(neurons))
        neurons.append(neuron)
    
    for idx in range(Nb):
         neuron = {
            "type" : "FS_Neuron",
            "compartments" : basket_fs_neuron.copy()
         }
         neuron["compartments"]["V0"] += 10 * np.random.rand()
         neuron["compartments"]["Iextmean"] += 0.5 * np.random.randn()
         indexes["bas"].append(len(neurons))
         neurons.append(neuron)
         
    for idx in range(Nolm):
        neuron = {
            "type" : "olm_cell",
            "compartments" : olm_params.copy()
        }
        neuron["compartments"]["V0"] += 10 * np.random.rand()
        neuron["compartments"]["Iextmean"] += 0.5 * np.random.randn()
        indexes["olm"].append(len(neurons))
        neurons.append(neuron)
    
    for idx in range(Npv1):
       
        neuron = {
            "type" : "CosSpikeGenerator", 
            "compartments" : CosSpikeGeneratorParams.copy()
        }
        # neuron["compartments"]["threshold"] = 0.6
        neuron["compartments"]["phase"] = params["ms_pv1_pv2_phase_shift"] + params["ms_cs_pp_phase_shift"] 
        indexes["pv1"].append(len(neurons))
        neurons.append(neuron)
    
    for idx in range(Npv2):
        
        neuron = {
            "type" : "CosSpikeGenerator", 
            "compartments" : CosSpikeGeneratorParams.copy()
        }
    
        neuron["compartments"]["phase"] = params["ms_cs_pp_phase_shift"] 
        indexes["pv2"].append(len(neurons))
        
  
    for idx in range(Nec):    
        neuron = {
            "type" : "CosSpikeGenerator", 
            "compartments" : CosSpikeGeneratorParams.copy()
        }
        neuron["compartments"]["threshold"] = 0.3 
        neuron["compartments"]["phase"] = -2.79 # !!!!!!!!
        indexes["mec"].append(len(neurons))
        neurons.append(neuron)
    
    for idx in range(Ncs):    
        neuron = {
            "type" : "CosSpikeGenerator", 
            "compartments" : CosSpikeGeneratorParams.copy()
        }
        neuron["compartments"]["threshold"] = 0.3
        neuron["compartments"]["phase"] = 0 # !!!!!!!! 
        indexes["cs"].append(len(neurons))
        neurons.append(neuron)
    
    for idx in range(Ns):
        #synapse beteween pyramides 
        
        pre_ind = np.random.choice(indexes["pyr"])
        post_ind = np.random.choice(indexes["pyr"])
        
        if (pre_ind == post_ind):
            post_ind = np.random.randint(0, Np)
        synapse = {
           "type" : "ComplexSynapse", # "SimpleSynapseWithDelay",
           "pre_ind": pre_ind, 
           "post_ind": post_ind,
           "pre_compartment_name": "soma",
           "post_compartment_name" : "soma",
           "params" : ext_synapse_params.copy(),
        }
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
    
    for _ in range(Nbasket2pyr):
        for idx in indexes["pyr"]:
            # synapses from basket cells to pyramides
            
            pre_ind = np.random.choice(indexes["bas"])
            synapse = {
               "type" : "ComplexSynapse", # "SimpleSynapseWithDelay",
               "pre_ind": pre_ind, 
               "post_ind": idx,
               "pre_compartment_name": "soma",
               "post_compartment_name" : "soma",
               "params": inh_synapse_params.copy(),
            }
            synapse["params"]["delay"] = np.random.randint(20, 50)
            synapses.append(synapse)
    
    for _ in range(Nolm2pyr):
        for idx in indexes["pyr"]:
            # synapses from OLM cells to pyramides
            pre_ind = np.random.choice(indexes["olm"])
            
            synapse = {
               "type" : "ComplexSynapse", # "SimpleSynapseWithDelay",
               "pre_ind": pre_ind, 
               "post_ind": idx,
               "pre_compartment_name": "soma",
               "post_compartment_name" : "dendrite",
               "params": inh_synapse_params.copy(),
            }
            synapse["params"]["delay"] = np.random.randint(20, 50)
            synapses.append(synapse)
    
    
    
    for _ in range(Nspyr2OLM): 
        for idx in indexes["pyr"]:
            pre_ind = idx
            
            # set synapse from pyramide to OLM neuron
            post_ind = np.random.choice(indexes["olm"])
            synapse = {
               "type" : "ComplexSynapse", # "SimpleSynapseWithDelay",
               "pre_ind": pre_ind, 
               "post_ind": post_ind,
               "pre_compartment_name": "soma",
               "post_compartment_name" : "soma",
               "params": ext_synapse_params.copy(),
               
            }
            synapse["params"]["delay"] = np.random.randint(20, 50)
            synapses.append(synapse)
    
    for _ in range(Nspyr2basket):
        for idx in indexes["pyr"]:
            # set synapse from pyramide to basket neuron
            pre_ind = idx
            post_ind =  np.random.choice(indexes["bas"])
            synapse = {
               "type" : "ComplexSynapse", # "SimpleSynapseWithDelay",
               "pre_ind": pre_ind, 
               "post_ind": post_ind,
               "pre_compartment_name": "soma",
               "post_compartment_name" : "soma",
               "params": ext_synapse_params.copy(),
               
            }
            synapse["params"]["delay"] = np.random.randint(20, 50)
            synapses.append(synapse)
    
    
    
    
    for idx in range(Nspv12bas):
        # set synapses from pv1 to basket
        pre_ind =  np.random.choice(indexes["pv1"])
        post_ind = np.random.choice(indexes["bas"])
        synapse = {
            "type" : "SimpleSynapseWithDelay",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": simple_inh_synapse_params.copy(),
            }
        # synapse["params"]["Erev"] = -75
        # synapse["params"]["w"] = 100
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
    
    
    for idx in range(Nspv22olm):
        # set synapses from pv1 to olm
        pre_ind = np.random.choice(indexes["pv2"])
        post_ind = np.random.choice(indexes["olm"])
        synapse = {
            "type" : "SimpleSynapseWithDelay",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": simple_inh_synapse_params.copy()
        }
        # synapse["params"]["Erev"] = -75
        # synapse["params"]["w"] = 100
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
    
    

    for idx in range(NsCs2pyr):
        # synapses from shaffer colletarel to pyramodes
        pre_ind = np.random.choice(indexes["cs"])
        post_ind = np.random.choice(indexes["pyr"])
        synapse = {
            "type" : "SimpleSynapseWithDelay",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": simple_ext_synapse_params.copy()
        }

        # synapse["params"]["w"] = 20
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
    
    
    for idx in range(NsCs2bas):
        # synapses from shaffer colletarel to basket
        pre_ind = np.random.choice(indexes["cs"])
        post_ind = np.random.choice(indexes["bas"])
        synapse = {
            "type" : "SimpleSynapseWithDelay",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "soma",
            "params": simple_ext_synapse_params.copy()
        }

        #synapse["params"]["w"] = 100
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
    
    for idx in range(NsEc2pyr):
        # synapses from perforant pathway to pyramodes
        pre_ind = np.random.choice(indexes["mec"])
        post_ind = np.random.choice(indexes["pyr"])
        synapse = {
            "type" : "SimpleSynapseWithDelay",
            "pre_ind": pre_ind, 
            "post_ind": post_ind,
            "pre_compartment_name": "soma",
            "post_compartment_name" : "dendrite",
            "params": simple_ext_synapse_params.copy()
        }

        #synapse["params"]["w"] = 100
        synapse["params"]["delay"] = np.random.randint(20, 50)
        synapses.append(synapse)
    
    
    """
    for syn in synapses:
        print (neurons[syn["pre_ind"]]["type"] + " -> " + neurons[syn["post_ind"]]["type"])
        print ( str(syn["pre_ind"]) + " -> " + str(syn["post_ind"]) )
    return 
    """
    
    dt = 0.1
    duration = 500
    net = lib.Network(neurons, synapses)
    
    for _ in range(4):
        
        net.integrate(dt, duration)
        net.save_results(path + "_all_results")
 
        print ("Что-то посчиталось!!!")
    return indexes
#######################################################
saving_fig_path = "/home/ivan/Data/modeling_septo_hippocampal_model/hippocampal_model/"

params = {
    "ms_cs_pp_phase_shift" : 1.5,
    "ms_pv1_pv2_phase_shift" : -2.15,
    "n_pv1" : 50,
    "n_pv2" : 50,
    "n_ec" : 100,
    "n_cs" : 100,
    "Nspv12bas" : 400,
    "Nspv22olm" : 200,
    "NsEc2pyr" : 25,       
    "NsCs2pyr" : 150,
    "NsCs2bas" : 25,
    "Nsbas2pyr" : 10,
    "Nsolm2pyr"  : 10,
}



path = saving_fig_path + "bas_inputs_density/"
if not( os.path.isdir(path) ):
    os.mkdir(path)

number_synapses = np.repeat( np.arange(0, 110, 10, dtype=int), 10)
    
for idx, ns in enumerate(number_synapses):
    path_tmp = path + str(idx + 1) + "_"
    params_tmp = params.copy()

    params_tmp["Nsbas2pyr"] = ns
    
    t = time.time()
    indexes = run_model(path_tmp, params_tmp)
    print (time.time() - t)
    
######################
path = saving_fig_path + "olm_inputs_density/"
if not( os.path.isdir(path) ):
    os.mkdir(path)    

for idx, ns in enumerate(number_synapses):
    path_tmp = path + str(idx + 1) + "_"
    params_tmp = params.copy()

    params_tmp["Nsolm2pyr"] = ns
    
    t = time.time()
    indexes = run_model(path_tmp, params_tmp)
    print (time.time() - t)



# import lfp_processing

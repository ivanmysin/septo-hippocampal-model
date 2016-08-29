# -*- coding: utf-8 -*-
import ca1
import numpy as np
import matplotlib.pyplot as plt

olm_params = {
    "V0": -60,
    "C" : 1.0,
    "Iextmean": -9.5,        
    "Iextvarience": 0.8,
    "ENa": 55.0,
    "EK": -90.0,
    "El": -65.0,
    "EH": -20, 
    "gbarNa": 52.0,
    "gbarNap": 0.5,
    "gbarK_DR": 11.0,
    "gbarH": 1.45,
    "gl": 0.5,
}


olm = ca1.OLMcell(olm_params)
olm.integrate(0.01, 1000)
v = np.array (olm.getVhist())
t = np.linspace(0, 1000, v.size)
plt.plot(t, v)
# -*- coding: utf-8 -*-
"""
cython library
"""
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
cimport numpy as np
import numpy as np


cdef extern from "Neurons.h" namespace "complex_neurons":
    cdef cppclass Compartment:
        Compartment(map[string, double]) except +
        vector [double] getVhist()
        double getV()
        void integrate(double, double)

        
cdef class pyCompartment:
    # cdef Compartment comp  
    # hold a C++ instance which we're wrapping
    cdef Compartment* comp 
    def __cinit__(self, params_):
        cdef map[string, double] params
        cdef pair[string, double] p
        
        for key, value in params_.items():
            p.first = key.encode("UTF-8")
            p.second = value
            params.insert(p)
        
        self.comp = new Compartment(params)
        
    def integrate(self, dt, duration):
        self.comp.integrate(dt, duration)
        
    def getVhist(self):
        cdef vector[double] Vhist = self.comp.getVhist()
        N = Vhist.size()
        Vhist_ = np.empty(N, dtype=np.float64)
        for idx in range(N):
            Vhist_[idx] = Vhist[idx]
        return Vhist_
        
    def getVhistbyIdx(self, idx):
        cdef vector[double] Vhist = self.comp.getVhist()
        for i in range (Vhist.size()):
            print (Vhist[i])
        return Vhist[idx]
    
    def getNVhistvalues(self):
        return self.comp.getVhist().size()

    def getV(self):
        return self.comp.getV()
            
    def test(self):
        print ("hello")
      

    
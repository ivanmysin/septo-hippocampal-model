# -*- coding: utf-8 -*-
"""
cython library
"""
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
cimport numpy as np



cdef extern from "Neurons.h" namespace "complex_neurons":
    cdef cppclass Compartment:
        Compartment() except +
        Compartment(map[string, double]) except +
        Compartment(Compartment  &) except +
        vector [double] getVhist()
        double getV()
        void integrate(double, double)
    cdef cppclass Connection:
        Connection(Compartment* compartment1_, Compartment* compartment2_, double g_, double p_) except +
        void activate()
        
    cdef cppclass ComplexNeuron:
        ComplexNeuron() except +
        ComplexNeuron(map[string, Compartment*]*, map[string, Connection*]*) except +
        void addCompartment(string, Compartment*)
        void addConnection(string, Connection*)
        void integrate(double, double)
        Compartment* getCompartmentbyName(string)
        vector [double] getVhistByCompartmentName (string)
        
        
        
cdef extern from "Neurons.h" namespace "synapses_models":
    cdef cppclass OriginSynapse:
        OriginSynapse()  except +
        void integrate(double)
    cdef cppclass SimpleSynapse:
        SimpleSynapse() except +
        SimpleSynapse(Compartment*, Compartment*, map[string, double]) except +
        void integrate(double)
    
   
cdef extern from "Neurons.h" namespace "networks":
    cdef cppclass Network:
        Network()
        Network(vector [ComplexNeuron*]*, vector [OriginSynapse*]*)
        void init()
        void integrate(double, double)
        void addNeuron(ComplexNeuron*)
        void addSynapse(SimpleSynapse*)

        
        
cdef class pyNetwork:
    #  cdef Network* net
    cdef vector [ComplexNeuron*]* neurons 
    cdef vector [SimpleSynapse*]* synapses 
   
    def __cinit_(self, neurons_params, synapses_params):
        
        
        # self.net = new Network()
        self.neurons = new vector [ComplexNeuron*]()
        self.synapses = new vector [SimpleSynapse*]()
        
        # ComplexNeuron(map[string, Compartment*]*, map[string, Connection*]*)
        cdef string compartment_name
        
        cdef pyComplexNeuron neuron
        cdef SimpleSynapse* synapse
        #for one_neuron in neurons_params:
        neuron = pyComplexNeuron(neurons_params[0]["compartments"], neurons_params[0]["connections"])
        # print (neuron.getVhistByCompartmentName("soma"))
        self.neurons.push_back( neuron.getCppObj() )
                
        cdef map[string, double] syn_params
        cdef pair [string, double] syn_params_pair
        cdef Compartment* pre
        cdef Compartment* post
        cdef int idx = 0
        for idx in range(len(synapses_params)):
            pre  = self.neurons.at(synapses_params[idx]["pre_ind"]).getCompartmentbyName(synapses_params[idx]["pre_compartment_name"].encode("UTF-8"))
            post = self.neurons.at(synapses_params[idx]["post_ind"]).getCompartmentbyName(synapses_params[idx]["post_compartment_name"].encode("UTF-8"))
            for key, value in synapses_params[idx]["params"].items():
                syn_params_pair.first = key.encode("UTF-8")
                syn_params_pair.second = value
                syn_params.insert(syn_params_pair)
            synapse = new SimpleSynapse(pre, post, syn_params)
            self.synapses.push_back(synapse)
            
    def integrate(self, double dt, double duration):
        #self.net.init()
        #self.net.integrate(dt, duration)
        print ("Hello")
        
cdef class pyCompartment:
    # cdef Compartment comp  
    # hold a C++ instance which we're wrapping
    cdef Compartment* comp 
    # cdef Compartment* getCppObj()
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
        
    def getV(self):
        return self.comp.getV()
        
    cdef Compartment* getCppObj(self):
        return self.comp
          
    def test(self):
        print ("hello")
      
cdef class pyConnection:
    cdef Connection* conn 
    def __cinit__(self, pyCompartment comp1, pyCompartment comp2, double g, double p):
        self.conn = new Connection( comp1.getCppObj(), comp2.getCppObj(), g, p )
    cdef activate(self):
        self.conn.activate()
        

cdef class pyComplexNeuron:
    cdef ComplexNeuron* neuron 
    def __cinit__(self, list compartments, list connections):
        
        self.neuron = new ComplexNeuron()
          
        cdef map[string, double] params
        cdef pair[string, double] p        
        cdef string name 
        
        for comp in compartments:
            for key, value in comp["params"].items():
                p.first = key.encode("UTF-8")
                p.second = value
                params.insert(p)
            
     
            name = comp["name"].encode("UTF-8")
            pcomp = new Compartment(params)
            self.neuron.addCompartment(name, pcomp)
            params.clear()

        cdef string comp1_name
        cdef string comp2_name
        cdef Connection* pconn
        
        for conn in connections:
            
            comp1_name = conn["compartment1"].encode("UTF-8");
            comp2_name = conn["compartment2"].encode("UTF-8");
            
            pconn = new Connection(self.neuron.getCompartmentbyName(comp1_name), self.neuron.getCompartmentbyName(comp2_name) , conn["g"], conn["p"])
            name = conn["name"].encode("UTF-8")          
            self.neuron.addConnection(name, pconn)
        
    
        
    def integrate(self, double dt, double duration):
        self.neuron.integrate(dt, duration)    
        
    def getVhistByCompartmentName(self, name):
        cdef string namecpp = name.encode("UTF-8")
        cdef vector [double] Vhist = self.neuron.getVhistByCompartmentName(namecpp)
        N = Vhist.size()
        Vhist_ = np.empty(N, dtype=np.float64)
        for idx in range(N):
            Vhist_[idx] = Vhist[idx]
        return Vhist_
    
    cdef ComplexNeuron* getCppObj(self):
        return self.neuron

        
        
        
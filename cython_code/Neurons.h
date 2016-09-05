
#include <math.h>
#include <map>
#include <string>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

namespace complex_neurons {
    class OriginCompartment{
    public:
        OriginCompartment(){};
        ~OriginCompartment(){};
        double getV(){return V;};
        void setIext(double Iext_){Iext = Iext_;};
        void addIsyn(double Isyn_){Isyn += Isyn_;};
        vector <double> getVhist(){return Vhist;};
        virtual void integrate(double dt, double duration){};
    protected:
        double V, Isyn, Iext;
        vector <double> Vhist;

    };

    class Compartment:public OriginCompartment {
    public:
        Compartment(){};
        Compartment(map <string, double> params);
        Compartment(const Compartment &obj);
        ~Compartment();
        void integrate(double dt, double duration);
        void checkFired(double t_);
        double getV();
        vector<double> getVhist();
        // void addIsyn(double Isyn);
        void setIext(double Iext);
        vector<double>  getFiring();
		map <string, double> getParameters();
    protected:
		double V, Capacity, Iextmean, Iextvarience, ENa, EK, El, ECa, CCa, sfica, sbetaca;
		double gbarNa, gbarK_DR, gbarK_AHP, gbarK_C, gl, gbarCa;
		double th;
		bool countSp;
		double m, h, n, s, c, q;
		double Iext, Isyn, INa, IK_DR, IK_AHP, IK_C, ICa, Il;
		vector <double> Vhist;
		vector <double> firing;

        default_random_engine* generator;
        normal_distribution<double>* normRand;


        void calculate_currents();
		double alpha_m();
		double beta_m();
        double alpha_h();
        double beta_h();
        double alpha_n();
        double beta_n();
        double alpha_s();
        double beta_s();
        double alpha_c();
        double beta_c();
        double alpha_q();
        double beta_q();
        double h_integrate(double dt);
        double n_integrate(double dt);
        double s_integrate(double dt);
        double c_integrate(double dt);
        double q_integrate(double dt);
        double CCa_integrate(double dt);

    };
    ///////////////////////////////////////////////////////////
    class Connection {
	public:
		Connection(Compartment* compartment1_, Compartment* compartment2_, double g_, double p_);

		~Connection();
		void activate();
	protected:
		Compartment* compartment1;
		Compartment* compartment2;
		double g, p;
	};
    ///////////////////////////////////////////////////////////
    class ComplexNeuron {

	public:
		ComplexNeuron(){
            compartments = new map<string, Compartment*>();
            connections = new map<string, Connection*>();
		};
		void addCompartment(string name, Compartment* newCompartment) {
            compartments->insert(pair<string, Compartment*> (name, newCompartment) );
		};
        void addConnection(string name, Connection* newConnection) {
            connections->insert(pair<string, Connection*> (name, newConnection) );
		};
		ComplexNeuron(map<string, Compartment*>* compartments_, map<string, Connection*>* connections_);
		~ComplexNeuron();
		void integrate(double, double);
		Compartment* getCompartmentbyName(string);
		vector <double> getVhistByCompartmentName(string name);
	protected:
		map<string, Compartment*>* compartments;
		map<string, Connection*>* connections;
	};
};

namespace synapses_models {
    class OriginSynapse {
    public:

        OriginSynapse(){};
        void integrate(double){};
        virtual ~OriginSynapse(){};
    protected:
        complex_neurons::OriginCompartment* pre,* post;
        double W;
    };

    class SimpleSynapse:public OriginSynapse {
    public:
        SimpleSynapse(complex_neurons::OriginCompartment* pre_, complex_neurons::OriginCompartment* post_, map<string, double> params);
        ~SimpleSynapse(){};
        void integrate(double);
    protected:
        double Erev, tau, S, gbarS;
        

    };

}

namespace networks {
    class Network{
    public:
        Network(){ 
			cout << "Tsst 1" << endl;
            neurons = new vector<complex_neurons::ComplexNeuron*>();
            synapses = new vector<synapses_models::OriginSynapse*>();
        };
        Network( vector<complex_neurons::ComplexNeuron*>*, vector<synapses_models::OriginSynapse*>*   );
        ~Network(){};
        void integrate(double, double);
        void init();
        void addNeuron(complex_neurons::ComplexNeuron* newNeuron){neurons->push_back(newNeuron);};
        void addSynapse(synapses_models::OriginSynapse* newSynapse){ synapses->push_back(newSynapse); };
    private:
        vector<complex_neurons::ComplexNeuron*>* neurons;
        vector<synapses_models::OriginSynapse*>* synapses;
    };

}


#include "Neurons.h"

namespace complex_neurons {



    Compartment::Compartment(map <string, double> params) {

        V = params["V0"];
        Capacity = params["C"];
        Iextmean = params["Iextmean"];
        Iextvarience = params["Iextvarience"];

        ENa = params["ENa"];
        EK = params["EK"];
        El = params["El"];
        ECa = params["ECa"];

        CCa = params["CCa"];
        sfica = params["sfica"];
        sbetaca = params["sbetaca"];

        gbarNa = params["gbarNa"];
        gbarK_DR = params["gbarK_DR"];

        gbarK_AHP = params["gbarK_AHP"];
        gbarK_C = params["gbarK_C "];

        gl = params["gl"];
        gbarCa = params["gbarCa"];

        th = -20;
        countSp = false;

        m = alpha_m() / (alpha_m() + beta_m());
        h = alpha_h() / (alpha_h() + beta_h());
        n = alpha_n() / (alpha_n() + beta_n());
        s = alpha_s() / (alpha_s() + beta_s());
        c = alpha_c() / (alpha_c() + beta_c());
        q = alpha_q() / (alpha_q() + beta_q());

        unsigned int seed = chrono::system_clock::now().time_since_epoch().count();
        generator =  new default_random_engine (seed);
        normRand = new normal_distribution <double> (Iextmean, Iextvarience );


        calculate_currents();
    }



	Compartment::Compartment(const Compartment &obj){
		cout << "Hello from copy constructor" << endl;
		/*
		map <string, double> params = obj.getParameters();

		V = params["V0"];
        Capacity = params["C"];
        Iextmean = params["Iextmean"];
        Iextvarience = params["Iextvarience"];

        ENa = params["ENa"];
        EK = params["EK"];
        El = params["El"];
        ECa = params["ECa"];

        CCa = params["CCa"];
        sfica = params["sfica"];
        sbetaca = params["sbetaca"];

        gbarNa = params["gbarNa"];
        gbarK_DR = params["gbarK_DR"];

        gbarK_AHP = params["gbarK_AHP"];
        gbarK_C = params["gbarK_C "];

        gl = params["gl"];
        gbarCa = params["gbarCa"];

        th = -20;
        countSp = false;

        m = alpha_m() / (alpha_m() + beta_m());
        h = alpha_h() / (alpha_h() + beta_h());
        n = alpha_n() / (alpha_n() + beta_n());
        s = alpha_s() / (alpha_s() + beta_s());
        c = alpha_c() / (alpha_c() + beta_c());
        q = alpha_q() / (alpha_q() + beta_q());

        calculate_currents();*/
	}



    map <string, double> Compartment::getParameters(){

		map <string, double> params;
		params["V0"] = V;
		params["C"] = Capacity;
		params["Iextmean"] = Iextmean;
                                 params[ "Iextvarience"] = Iextvarience;
 								 params[ "ENa"] = ENa;
                                 params[ "EK"] = EK;
                                 params[ "El"] = El;
                                 params[ "ECa"] = ECa;

                                 params[ "CCa"] = CCa;
                                 params["sfica"] = sfica;
                                 params["sbetaca"] = sbetaca;
                                 params["gbarNa"] = gbarNa;

                                 params["gbarK_DR"] = gbarK_DR;
                                 params["gbarK_AHP"] = gbarK_AHP;
                                 params["gbarK_C"] = gbarK_C;

                                 params["gl"] = gl;
                                 params["gbarCa"] = gbarCa;
		/* = {
								 make_pair( "V0", V ),
                                 make_pair( "C", Capacity ),
                                 make_pair( "Iextmean", Iextmean ),
                                 make_pair( "Iextvarience", Iextvarience),
 								 make_pair( "ENa", ENa ),
                                 make_pair( "EK", EK ),
                                 make_pair( "El", El ),
                                 make_pair( "ECa", ECa),

                                 make_pair( "CCa", CCa),
                                 make_pair("sfica", sfica),
                                 make_pair("sbetaca", sbetaca),
                                 make_pair("gbarNa", gbarNa),

                                 make_pair("gbarK_DR", gbarK_DR),
                                 make_pair("gbarK_AHP", gbarK_AHP),
                                 make_pair("gbarK_C", gbarK_C),

                                 make_pair("gl", gl),
                                 make_pair("gbarCa", gbarCa),


                                };
                                */
		return params;
	}

/////////////////////////////////////////////
    void Compartment::calculate_currents(){
        Il = gl * (V - El);
        INa = gbarNa * m * m * h * (V - ENa);
        IK_DR = gbarK_DR * n * (V - EK);
        IK_AHP = gbarK_AHP * q * (V - EK);
        IK_C = gbarK_C * c * (V - EK);

        double tmp = CCa / 250.0;
        if (tmp < 1) {
            IK_C *= tmp;
        }

        ICa = gbarCa * s * s * (V - ECa);
        // Iext = Iextmean;
        Iext = (*normRand)(*generator);
        Isyn = 0;
    }
///////////////////////////////////////////
    double Compartment::alpha_m(){
        double x = 13.1 - V;
        if (x == 0){
            x = 0.000001;
        };
        double alpha = 0.32 * x / (exp(0.25 * x) - 1);
        return alpha;
    }
///////////////////////////////////////////
	double Compartment::beta_m(){
        double x = V - 40.1;
        if (x == 0){
            x = 0.00001;
        }
        double beta = 0.28 * x / (exp(0.2 * x) - 1);
        return beta;
    }
//////////////////////////////////////////
    double Compartment::alpha_h(){
        double alpha = 0.128 * exp((17 - V) / 18);
        return alpha;
    }
//////////////////////////////////////////
    double Compartment::beta_h(){
        double x = 40 - V;
        if (x == 0){
            x = 0.00001;
        }
        double beta = 4 / (exp(0.2 * x) + 1);
        return beta;
	}
//////////////////////////////////////////
    double Compartment::alpha_n(){
        double x = 35.1 - V;
        if (x == 0) {
            x = 0.00001;
        }
        double alpha = 0.016 * x / (exp(0.2 * x) - 1);
        return alpha;
	}
//////////////////////////////////////////
	double Compartment::beta_n(){
        double beta = 0.25 * exp(0.5 - 0.025 * V);
        return beta;
    }
///////////////////////////////////////////
	double Compartment::alpha_s(){
        double x = V - 65;
        double alpha = 1.6 / (1 + exp(-0.072 * x));
        return alpha;
    }
///////////////////////////////////////////
    double Compartment::beta_s(){
        double x = V - 51.1;
        if (x == 0){
            x = 0.00001;
        };
        double beta = 0.02 * x / (exp(0.2 * x) - 1);
        return beta;
	}
///////////////////////////////////////////
    double Compartment::alpha_c(){
		double alpha;
        if(V > 50){
            alpha = 2 * exp((6.5 - V)/27);
        } else {
            alpha = exp( ((V - 10)/11) - ((V - 6.5)/27) ) / 18.975;
        }
        return alpha;
    }
//////////////////////////////////////////
    double Compartment::beta_c(){
		double beta;
        if (V > 0){
            beta = 0;
        } else {
            beta = 2 * exp((6.5 - V)/27) - alpha_c();
        }
        return beta;
    }
/////////////////////////////////////////
    double Compartment::alpha_q(){
        double alpha = 0.00002 * CCa;
        if (alpha > 0.01){
            alpha = 0.01;
        }
        return alpha;
    }
////////////////////////////////////////
    double Compartment::beta_q(){
        return 0.001;
    }
////////////////////////////////////////
    double Compartment::h_integrate(double dt){
        double h_0 = alpha_h() / (alpha_h() + beta_h());
        double tau_h = 1 / (alpha_h() + beta_h());
        return h_0 - (h_0 - h) * exp(-dt / tau_h);
	}
///////////////////////////////////////
    double Compartment::n_integrate(double dt){
        double n_0 = alpha_n() / (alpha_n() + beta_n() );
        double tau_n = 1 / ( alpha_n() + beta_n() );
        return n_0 - (n_0 - n) * exp(-dt / tau_n);
    }
////////////////////////////////////////
    double Compartment::s_integrate(double dt) {
        double s_0 = alpha_s() / (alpha_s() + beta_s() );
        double tau_s = 1 / (alpha_s() + beta_s());
        return s_0 - (s_0 - s) * exp(-dt / tau_s);
    }
///////////////////////////////////////
    double Compartment::c_integrate(double dt){
        double c_0 = alpha_c() / (alpha_c() + beta_c() );
        double tau_c = 1 / (alpha_c() + beta_c());
        return c_0 - (c_0 - c) * exp(-dt / tau_c);
     }
//////////////////////////////////////////////
    double Compartment::q_integrate(double dt){
        double q_0 = alpha_q() / (alpha_q() + beta_q() );
        double tau_q = 1 / (alpha_q() + beta_q());
        return q_0 - (q_0 - q) * exp(-dt / tau_q);
    }
///////////////////////////////////////////////
    double Compartment::CCa_integrate(double dt){
        double k1 = CCa;
        double k2 = k1 + 0.5 * dt * (-sfica * ICa - sbetaca * k1);
        double k3 = k2 + 0.5 * dt * (-sfica * ICa - sbetaca * k2);
        double k4 = k1 + dt * (-sfica * ICa - sbetaca * k1);
        return (k1 + 2*k2 + 2*k3 + k4) / 6.0;
    }
/////////////////////////////////////////////////
    void Compartment::integrate(double dt, double duration){
        double t = 0;
        while (t < duration){
			Vhist.push_back(V);
            V += dt * (-Il - INa - IK_DR - IK_AHP - IK_C - ICa - Isyn + Iext) / Capacity;
            m = alpha_m() / (alpha_m() + beta_m());
            h = h_integrate(dt);
            n = n_integrate(dt);
            s = s_integrate(dt);
            c = c_integrate(dt);
            q = q_integrate(dt);
            CCa = CCa_integrate(dt);
            calculate_currents();

            t += dt;
        }
    }
///////////////////////////////////////////////
    void Compartment::checkFired(double t_){

        if (V >= th && countSp){
            firing.push_back(t_);
            countSp = false;
        }
        if (V < th){
            countSp = true;
        }
	}
//////////////////////////////////////////////////
    double Compartment::getV(){
        return V;
    }

    vector <double> Compartment::getVhist(){
        return Vhist;
    }

    //void  Compartment::addIsyn(double Isyn_){
    //    Isyn += Isyn_;
    //}

    void Compartment::setIext(double Iext_){
        Iext = Iext_;
    }

    vector <double> Compartment::getFiring(){
        return firing;
    }

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
	Connection::Connection(Compartment* compartment1_, Compartment* compartment2_, double g_, double p_) {
		compartment1 = compartment1_;
		compartment2 = compartment2_;
		g = g_;
		p = p_;
	}

	Connection::~Connection(){}

	void Connection::activate(){
		double Icom1 = (g / p) * ( compartment1->getV() - compartment2->getV() );
        double Icom2 = (g/(1 - p)) * ( compartment2->getV() - compartment1->getV() );

        // cout << compartment1->getV() << "  " <<  compartment2->getV() << endl;
        compartment1->addIsyn(Icom1);
        compartment2->addIsyn(Icom2);
	}
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
	ComplexNeuron::ComplexNeuron(map<string, Compartment*>* compartments_, map<string, Connection*>* connections_){
		compartments = compartments_;
		connections = connections_;
	}
	ComplexNeuron::~ComplexNeuron(){}

	void ComplexNeuron::integrate(double dt, double duration){
        double t = 0;
        while (t < duration){
            for (map<string, Compartment*>::iterator it = compartments->begin(); it != compartments->end(); it++) {
                it->second->integrate(dt, dt);
            }

            for (map<string, Connection*>::iterator it = connections->begin(); it != connections->end(); it++) {
               it->second->activate();
            }

            t += dt;

		}
	}

	Compartment* ComplexNeuron::getCompartmentbyName(string name){
		return compartments->at(name);
	}

	vector <double> ComplexNeuron::getVhistByCompartmentName(string name){
		return compartments->at(name)->getVhist();
	}


}

namespace synapses_models {

/////////////////////////////////////////////////////////////////////
    SimpleSynapse::SimpleSynapse(complex_neurons::OriginCompartment* pre_, complex_neurons::OriginCompartment* post_, map<string, double> params){
        pre = pre_;
        post = post_;
        Erev = params["Erev"];
        tau = params["tau"];
        W = params["W"];
        gbarS = params["gbarS"];
        S = 0;
    };
////////////////
    void SimpleSynapse::integrate(double dt) {
        double Vpre = pre->getV();

        if (Vpre > 40){
             S = 1;
        }

        if ( S < 0.005 ){
            S = 0;
            return;
        }


        double Vpost = post->getV();
        double Isyn = W * gbarS * S * (Vpost - Erev);
        post->addIsyn(Isyn); //  Isyn for post neuron

        // usage Runge_Kutta 4-th order integration
        double k1 = S;
        double k2 = k1 - 0.5 * dt * (tau * k1);
        double k3 = k2 - 0.5 * dt * (tau * k2);
        double k4 = k1 - dt * (tau * k1);

        S = (k1 + 2*k2 + 2*k3 + k4) / 6.0;

    }

}

namespace networks {
    Network::Network( vector<complex_neurons::ComplexNeuron*>* neurons_, vector<synapses_models::OriginSynapse*>*  synapses_ ){
		cout << "Tsst 3" << endl;
        neurons = neurons_;
        synapses = synapses_;

    }
//////////////
    void Network::integrate(double dt, double duration){
        double t = 0;
		cout << "Tsst 2" << endl;
        while (t < duration) {
            for ( unsigned int i = 0; i < neurons->size(); i++ ) {
                neurons->at(i)->integrate(dt, dt);
            }

            for ( unsigned int i = 0; i < synapses->size(); i++  ) {
               synapses->at(i)->integrate(dt);
            }

            t += dt;
        }

    }
////////////
    void Network::init(){
			cout << "Test 1" << endl;
            vector<complex_neurons::ComplexNeuron*>* neu = new vector<complex_neurons::ComplexNeuron*> ();
            neurons = neu;
            //synapses = new vector<synapses_models::OriginSynapse*>();
			
	}
}





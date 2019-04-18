/* 
 * Copyright 2019 Jan Totz <janjan2.718128@googlemail.com>
 * 
 */


// C++11 & Boost libraries
#include <vector>								// std::vector
#include <iostream>								// std::cout, std::endl, std::ios::out, std::ios::binary, std::ios::app
#include <random>								// std::default_random_engine, std::normal_distribution<double>, std::uniform_real_distribution
#include <fstream>								// std::ofstream, std::ifstream
#include <chrono>								// std::chrono::high_resolution_clock::now()
#include <algorithm>							// std::max_element()


#include "./params.hpp"							// struct params

#ifdef GPU
	#include "./GPU/solver_interface_gpu.hu"				// solver();
#else
	#include "./CPU/solver_interface_cpu.hpp"				// solver();
#endif


#include "./Utilities/get_ini_params.hpp"		// organize_ini_options(); functions to read params from ini and cmdline
#include "./Utilities/initial_condition.hpp"	// initialCondition(); set initial condition




// save state as binary float
void saveParamDistro(Real *het, params &p){
	
	std::ofstream dataout;
	dataout.open(p.pthout+"/paramDistro.bin", std::ios::binary);
	
	// save header (1 int): nx
	dataout.write((char*) &p.nx, sizeof(int));
	
	// save data
	for(int n=0; n<p.n; n++) dataout.write((char*) &het[n], sizeof(Real));
	
	dataout.close();
}


void parameterDistribution(Real *hetArray, params &p, modelparams &m){
	
	std::default_random_engine generator(p.rngState + p.pSeed);
	
	switch(p.hehoflag){
		case 0:				// nothing; option for faster computation
			break;
		
		case 8:				// bounded Cauchy-Lorentz distribution
			std::cout << "parameterDistribution (" << p.hehoflag << "): bounded Cauchy-Lorentz distribution" << std::endl;
			{
				std::cauchy_distribution<double> p_cauchy_distribution(p.het1,p.pSigma);
				for(int i=0; i<p.n; i++){
					float prnd=p.het1+100*p.pSigma;
					while(prnd<p.het1-p.pSigma or prnd>p.het1+p.pSigma) prnd = p_cauchy_distribution(generator);			// limits for parameter p
					hetArray[i]=prnd;
				}
			}
			break;
		
		
		default:
			std::cout << "chosen value (" << p.hehoflag << ") for hehoflag is not implemented!" << std::endl; exit(1);
			break;
	}
	
	
	// save k array (contains heterogeneity distribution)
	saveParamDistro(hetArray,p);
	
	// DEBUG
	//~ for(int i=0; i<100; i++) cout << hetArray[i] << endl;
}



// translate array from (u1,v1,w1) -> (u1 u2 ... uN, ...) etc (= transpose)
// acts only on cAnalysis, not for array that is going to be saved!
// = transpose!
void translateArrayOrder(Real *cfield, Real *cAnalysis, params &p, int &untranslatedQ, int analysisArrayQ){
	
	if(analysisArrayQ){
		for(int c=0; c<2; c++){
		for(int i=0; i<p.n; i++){
			cAnalysis[i+c*p.n]=cfield[c+i*p.ncomponents];
		}}
		
	}else{
		switch(p.ncomponents){
			case 1:
				for(int i=0; i<p.n; i++) cAnalysis[i]=cfield[i];
				break;
				
			case 2:
				for(int c=0; c<p.ncomponents; c++){
				for(int i=0; i<p.n; i++){
					cAnalysis[i+c*p.n]=cfield[c+i*p.ncomponents];
				}}
				break;
				
			default:
				std::cout << "translateArrayOrder: number of components not supported!" << std::endl; exit(EXIT_FAILURE);
				break;
		}
	}
	untranslatedQ=0;
}






int main(int argumentsCount, char* argumentsValues[]){
	
	// message at program start
	std::cout << "############ Solver is starting ############ " << std::endl;
	
	// 0. parameter declaration
	struct params p;
	struct modelparams m;
	auto t1=std::chrono::high_resolution_clock::now(), t2=std::chrono::high_resolution_clock::now();
	
	
	// 1. get settings for simulation options
	organize_ini_options(argumentsCount,argumentsValues,p,m);
	
	// 2. prepare system
	const int nc = p.ncomponents==3 ? 4 : p.ncomponents;	// for speed trick of 3-component vectors on GPU, only applicable in cfield[...]
	Real *cfield = new Real[p.n*nc]{};						// dynamic variables = concentrations, {} = init with zeros
	Real *hetfield = new Real[p.n]{};						// parameter k for every cell
	initialCondition(cfield,p,m);							// set initial condition
	parameterDistribution(hetfield,p,m);					// set parameter distribution
	
	
	// 3. simulation
	// timer start
	t1 = std::chrono::high_resolution_clock::now();
	
	// simulation
	solver(cfield,hetfield,p,m);
	
	// timer end
	t2 = std::chrono::high_resolution_clock::now();
	int runtime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	
	// write report
	std::ofstream dataout;
	char spbuff[100];
	sprintf(spbuff,"execution time: %d ms = %d h, %d min, %d s.",runtime,runtime/1000/3600,runtime/1000/60%60,int(floor(runtime/1000.+0.5))%60);
	dataout.open(p.pthout+"/report.txt", std::ios::out);
	dataout << spbuff << std::endl;
	dataout.close();
	std::cout << "\n" << spbuff << std::endl;		// output to terminal
	
	// deallocate memory
	delete[] cfield;
	delete[] hetfield;
	delete[] m.coupling_coeffs;
	delete[] m.model_params;
	delete[] m.model_phases;
	
	return 0;
}


/* 
 * Copyright 2019 Jan Totz <jantotz@itp.tu-berlin.de>
 * 
 */


// C++11, OpenMP
#include <iostream>								// std::cout, std::cerr, std::endl
#include <fstream>								// std::ofstream, std::ifstream
#include <algorithm>							// std::transform()
#include <vector>								// std::vector
#include <cmath>								// exp()


#include "../params.hpp"						// struct params
#include "../Utilities/safe.hpp"				// saving
#include "./solver_cpu.hpp"						// struct host_pointers
#include "./models.hpp"							// reaction models
#include "./coupling.hpp"						// coupling schemes






Real kernelfunction(int i, int i0, params &p){
	
	Real value=0.0;
	
	switch(p.spaceDim){
		case 1:
			value=p.dx*exp(-abs(i-i0)*p.dx/p.kappa);
			break;
		default: printf("Error: spaceDim=%d not implemented.\n",p.spaceDim); exit(EXIT_FAILURE); break;
	}
	
	return value;
}


void create_kernel_and_rescale(int &maskSize, host_pointers *h, params &p, modelparams &m){
	
	switch(p.couplingChoice){
		
		// nonlocal BZ chimera coupling: Exp(-d/kappa), d = euclidean distance
		case 5:
		case 6:
			{
				int i=0, i0=0;
				p.kdia = 2*p.cutoffRange+1;
				p.kradius = p.cutoffRange;
				p.ksum = 0.0;
				switch(p.spaceDim){
					case 1:
						maskSize=p.kdia;
						i0=p.cutoffRange;
						h->mask = new Real[maskSize];
						for(i=0; i<p.kdia; i++) h->mask[i] = kernelfunction(i,i0,p);
						for(i=0; i<maskSize; i++) h->mask[i] *= p.K*p.dt;
						break;
					
					default: printf("Error: spaceDim=%d not implemented.\n",p.spaceDim); exit(EXIT_FAILURE); break;
				}
				// save kernel in binary data format for later
				std::ofstream dataout;
				dataout.open(p.pthout+"/coupling_kernel.bin", std::ios::binary);
				for(int i=0; i<maskSize; i++) dataout.write((char*) &(h->mask[i]), sizeof(Real));
				dataout.close();
				for(int i=0; i<maskSize; i++) p.ksum += h->mask[i];
				
			}
			break;
		
		
		// rescaling is accounted for in coupling
		case 1002:		// global coupling
		case 1003:		// global coupling with frustration
		case 1004:		// global coupling with delay
			break;
		
		default: printf("Unknown value for diffusionChoice (create_kernel_and_rescale()).\n"); exit(EXIT_FAILURE); break;
	}
}




void init_solver(params &p, modelparams &m, host_pointers *h){
	
	// initialize variables
	Real needed_mem=0.0;				// memory in bytes
	int array_size=p.n*p.ncomponents;	// size of 1 state
	
	
	// create kernel and rescale coupling coefficents
	int maskSize=0;
	create_kernel_and_rescale(maskSize,h,p,m);
	if( p.couplingChoice==5 || p.couplingChoice==6 ){
		printf("kernel size: %.3f MB\n",maskSize*sizeof(Real)/(1024.*1024.));
		needed_mem+=maskSize*sizeof(Real);
	}
	
	
	// delay and non-delay coupling arrays and pointers
	if(p.delayFlag==1){
		int history_array_size = array_size*(p.delayStepsMax+1);
		printf("c0 size: %.3f MB = states: %zu\n",history_array_size*sizeof(Real)/(1024.*1024.), p.delayStepsMax+1);
		h->c0 = new Real[history_array_size]();		// holds all concentration values for time delay (history array)
		
		for(int i=0; i<array_size; i++) h->c0[i] = h->c[i];					// set initial conditions
		// init array pointers for first step
		h->c = h->c0;
		h->cnew = h->c0 + array_size;
	}else if(p.delayFlag==0){
		h->cnew = new Real[array_size]();			// all dynamic variables {u,v,w,} (more are not implemented), init to zero
	}
	
	
	
}


void cleanup_CPU(host_pointers *h, params &p){
	
	// deallocate memory
	if(p.couplingChoice==5 or p.couplingChoice==6) delete[] h->mask;
	if(p.delayFlag==0){
		if(p.stepsEnd % 2 == 1) delete[] h->c;
		else delete[] h->cnew;
	}else if(p.delayFlag==1){
		delete[] h->c0;
	}
	
}



void time_evolution(host_pointers *h, params &p, modelparams &m, size_t step){
	
	switch(p.timeEvolutionChoice){
		case 0:										// explicit euler
			reaction(h->c,h->cnew,h,p);							// reaction part, see models.cpp
			coupling(h->c,h->cnew,h,p,m,step);					// spatial coupling, see coupling.cpp
			break;
		
	
		default: printf("Error: timeEvolutionChoice not implemented!"); exit(EXIT_FAILURE); break;
	}
}



void dynamics(host_pointers *h, params &p, modelparams &m, size_t step){
	
	//~ printf("dynamics\n");
	switch(p.delayFlag){
		case 0:											// no delay
			time_evolution(h,p,m,step);
			std::swap(h->c,h->cnew);					// update position values = pointer swap
			break;
		
		case 1:											// with delay
			time_evolution(h,p,m,step);
			
			// pointer position iteration as update
			h->c = h->c0 + ((step+1) % (p.delayStepsMax+1))*p.n*p.ncomponents;
			h->cnew = h->c0 + ((step+2) % (p.delayStepsMax+1))*p.n*p.ncomponents;
			
			if(step>=p.delayStartSteps){
				size_t delaySteps=0;
				if(p.delayTimeIntervalUntilMaxReached==0){		// const delay
					delaySteps=p.delayStepsMax;
				}else{											// linearly increasing delay
					delaySteps=(size_t)std::max((Real)p.delayStepsMax,p.delayStepsMin + p.dt*(p.delayTimeMax-p.delayTimeMin)/p.delayTimeIntervalUntilMaxReached*(step-p.delayStartSteps));
				}
				h->cdelay = h->c0 + ((step+3) % (delaySteps+1))*p.n*p.ncomponents;
			}
			
			break;
		
		default: printf("Unknown value for delayFlag (rd_dynamics).\n"); exit(EXIT_FAILURE); break;
	}
}



void solverCPU_network(Real *c, Real *het, params &p, modelparams &m){
	
	host_pointers h;
	h.c = c; h.het = het;
	init_solver(p,m,&h);
	
	// init classes for analysis and saving
	Safe safe(p);
	
	// save initial condition
	safe.save(h.c,0);
	
	
	// time loop
	for(size_t step=0; step<=p.stepsEnd; step++){
		
		// dynamics
		dynamics(&h,p,m,step);
		
		if(h.c[0]!=h.c[0]){ printf("step: %zu, u[0]=%f. Abort!\n",step,h.c[0]); exit(EXIT_FAILURE); }		// DEBUG
		
		
		// step counter output
		if(step>0){
			
			// output: save state
			if(!(step%p.stepsSaveState)){ 
				safe.save(h.c,step);						// save concentrations
			}
			
		}
	}
	
	// clean up data
	cleanup_CPU(&h,p);
}


void solverCPU_1d(Real *c, Real *het, params &p, modelparams &m){
	
	host_pointers h;
	h.c = c; h.het = het;
	init_solver(p,m,&h);
	
	// init classes for analysis and saving
	Safe safe(p);
	
	// save initial condition
	safe.save(h.c,0);
	
	// time loop
	for(size_t step=0; step<=p.stepsEnd; step++){
		
		// dynamics
		dynamics(&h,p,m,step);
		
		if(h.c[0]!=h.c[0]){ printf("step: %zu, u[0]=%f. Abort!\n",step,h.c[0]); exit(EXIT_FAILURE); }		// DEBUG
		
		
		// step counter output
		if(step>0){
			
			// output: save state
			if(!(step%p.stepsSaveState)){ 
				safe.save(h.c,step);						// save concentrations
			}
			
		}
	}
	
	// clean up data
	cleanup_CPU(&h,p);
}





void solver(Real *c, Real *k1, params &p, modelparams &m){
	
	//~ printf("solverCPU - serial version\n");
	
	switch(p.spaceDim){
		case 1:														// 1d
			solverCPU_1d(c,k1,p,m); 
			break;
		case 1000:													// network
			solverCPU_network(c,k1,p,m);
			break;
		default: printf("Error: spaceDim=%d not implemented!\n",p.spaceDim); exit(EXIT_FAILURE); break;
	}
}

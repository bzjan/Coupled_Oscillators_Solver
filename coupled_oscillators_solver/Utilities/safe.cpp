
#include <vector>								// std::vector
#include <fstream>								// std::ofstream
#include <iostream>								// std::cout, std::cerr, std::endl, std::ios::out, std::ios::binary, std::ios::app
#include <cmath>								// floor

#include "../params.hpp"					// struct params
#include "./safe.hpp"



// constructor
Safe::Safe(params &p_in){
	
	// initialize buffer to hold data and write it all periodically to a single output file
	p=p_in;																							// params
	float mem_buffer = p.n*p.ncomponents*p.nStatesSaveBuffer*sizeof(Real)/(1024.*1024.);			// amount of memory required for buffer
	float mem_system = 7500.0;																		// TODO: get from system
	printf("save buffer size: %.3f / %.3f MB\n",mem_buffer,mem_system);
	if(mem_buffer>mem_system){ printf("Error! More memory requested than available! (%.3f / %.3f)\n",mem_buffer,mem_system); exit(EXIT_FAILURE); }		// todo: abort with error, if more memory is requested than available!
	
	// initialize variables
	sprintf(spbuff,"/states/state.bin");				// name of output file, may be overwritten later in singleSave mode
	size=sizeof(Real);									// memory size of one array cell
	dn=1;												// data coarsening: 1,1+dn -> smaller filesize, right now: failure
	error = 0;
	counter = 0;
	write_counter = 0;
	
	size_t bufferSize=0;								// total size of buffer vector
	// round up for coarsening 5/2 -> 3: 1,3,5
	nx=(p.nx-1)/dn+1;
	bufferSize = p.n*p.ncomponents*p.nStatesSaveBuffer;
	if(!p.saveSingleQ) buffer.resize(bufferSize);
	//~ cout << "buffer size: " << buffer.size() << endl;
	//~ printf("save buffer size: %.3f MB\n",buffer.size()*sizeof(Real)/(1024.*1024.));
	
	dt=p.dt*p.stepsSaveState;
	
}


void Safe::save(Real *c, const size_t step){
	
	//~ cout << "save , " << counter << ", " << step << endl;
	
	if(p.saveSingleQ){								// special mode
		// save every output and create rendered image with mma; for immediate scrollring render and low file size pollution
		save_single(c);
	}else{												// default mode
		// always save data to buffer
		save_buffer(c);
		// save states periodically, internal counter for fast modulo replacement
		// or save at the last possibility
		if((step<p.stepsEnd-1 and counter==p.nStatesSaveBuffer) or step==((p.stepsEnd-1)/p.stepsSaveState)*p.stepsSaveState) save_file();
	}
}

void Safe::save_single(Real *c){
	
	sprintf(spbuff,"/states/state_%05d.bin",counter);
	dataout.open(p.pthout+spbuff,std::ios::binary);
	if(!dataout){ std::cerr << "Error: Can not open output file " << p.pthout+spbuff << std::endl; exit(EXIT_FAILURE); }		// error checking
	
	// save header (4 ints + 1 float=5*4 byte): nx,ny,nz,nc,dtt
	// space dimensions, number of components, and time discretization
	dataout.write((char*) &nx, sizeof(int));
	dataout.write((char*) &p.ncomponents, sizeof(int));
	dataout.write((char*) &dt, sizeof(float));				// # time scale: dt*saveSteps
	
	
	//~ for(int i=0; i<10; i++) cout << c[i] << " "; cout << endl;		// DEBUG
	
	// one complete concentration field after another:  (u1,u2,...uN, v1,v2,...vN)
	if(dn>1){										// coarsen output data
		for(int c0=0; c0<p.ncomponents; c0++){
		for(int x=0; x<p.nx; x+=dn){
			dataout.write((char*) &c[x + c0*p.n], size);
		}}
	}else{											// no coarsening of output data
		for(int i=0; i<p.n*p.ncomponents; i++) dataout.write((char*) &c[i], size);
	}
	
	dataout.close();
	
	counter++;
}


void Safe::save_buffer(Real *c){
	
	// save current concentration field to buffer for later file write
	for(int i=0; i<p.ncomponents*p.n; i++) buffer[i+counter*p.ncomponents*p.n] = c[i];
	
	counter++;
}

void Safe::save_file(){
	
	//~ cout << "save_file" << endl;
	
	// open, append to file
	dataout.open(p.pthout+spbuff, std::ios::out | std::ios::binary | std::ios::app);
	if(!dataout){ std::cerr << "Error: Can not open output file " << p.pthout+spbuff << std::endl; exit(EXIT_FAILURE); }		// error checking
	
	// only write header at first file write
	if(write_counter==0){
		// save header (4 ints + 1 float=5*4 byte): nx,ny,nz,nc,dtt
		// space dimensions, number of components, and time discretization
		dataout.write((char*) &nx, sizeof(int));
		dataout.write((char*) &p.ncomponents, sizeof(int));
		dataout.write((char*) &dt, sizeof(float));				// # time scale: dt*saveSteps
	}
	// DEBUG
	//~ for(int i=0; i<10; i++) cout << c[i] << " "; cout << endl;
	
	// save data to file
	//~ cout << "write data" << endl;
	//~ cout << "p.n: " << p.n << ", p.ncomponents: " << p.ncomponents << ", *: " << p.ncomponents*p.n << endl;
	//~ cout << "counter: " << counter << ", write_counter: " << write_counter << endl;
	for(int i=0; i<counter; i++){
	for(int j=0; j<p.ncomponents*p.n; j++){
		dataout.write((char*) &buffer[j+i*p.ncomponents*p.n], size);
	}}
	
	dataout.close();
	
	counter=0;
	write_counter++;
}



// C++11 & Boost libraries
#include <random>								// std::default_random_engine, std::uniform_real_distribution
#include <iostream>								// std::cout, std::cerr, std::endl
#include <fstream>								// std::ofstream, std::ifstream, std::ios::binary, std::ios::in
#include <sstream>								// std::stringstream

#include "../params.hpp"					// structs params, modelparams




inline double posi_fmod(const double i, const double n){ return fmod((fmod(i,n)+n),n); }
inline int posi_imod(const int i, const int n){ return (i % n + n) % n; }
Real phi(const int x, const float x0, const int y, const float y0){ return atan2(y-y0,x-x0); }								// complex phase
Real d(const int x1, const float x2, const int y1, const float y2){ return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)); }			// Euclidean distance



// get concentration to phase lookup map
// resize phase inside function
void get_limitCycleData(std::vector<Real>& phases, const int nc, const std::string pthfn_lc){ 
	
	std::ifstream inputFile;
	inputFile.open(pthfn_lc);
	if(!inputFile){ std::cerr << "Error: Can not open input file " << pthfn_lc << std::endl; exit(EXIT_FAILURE); }		// error checking
	std::string temp_line;
	int ni=0;
	while(getline(inputFile,temp_line)){
		if(!temp_line.empty()) ni++;
	}
	inputFile.clear();
	inputFile.seekg(inputFile.beg);
	//~ cout << "ni: " << ni << endl;		// DEBUG
	phases.resize(ni*nc);
	for(int i=0; i<ni; i++){
	for(int c=0; c<nc; c++){
		inputFile>>phases[c+i*nc];
	}}
	inputFile.close();
}







void ic_phase_gaussian_envelope(Real *cfield, params &p, const int nc, std::default_random_engine& g){
	printf("ic: %s\n",p.ic.c_str());
	
	if(p.spatial_noise_settings.size()==1) p.spatial_noise_settings={0,1}; 	// default settings, otherwise: ini settings
	
	if(p.spatial_noise_settings[1]>1){ printf("Error: spatial_noise_settings in ic_phase_gaussian_envelope must be in [0,1]\n"); exit(EXIT_FAILURE);}
	std::uniform_real_distribution<Real> udistribution(p.spatial_noise_settings[0],p.spatial_noise_settings[1]);
	
	
	// spatial_settings1 overloaded for options: spatial_settings1 = {xmean,ymean,zmean,xsigma,ysigma,zsigma}
	if(p.spatial_settings1.size()==1) p.spatial_settings1={0.5,0.1};			// default settings, otherwise: ini settings
	
	float mu_x = p.spatial_settings1[0]*p.nx;	// mean x
	float xsigma = p.spatial_settings1[1]*p.nx;	// variance x

	
	std::vector<Real> phases;				// initialize empty vector
	get_limitCycleData(phases,nc,p.pthfn_lc);
	int ni=phases.size()/p.ncomponents;
	
	
	for(int x=0; x<p.nx; x++){
		int index = exp(-0.25*(x-mu_x)*(x-mu_x)/(xsigma*xsigma) )*udistribution(g)*(ni-1);					// index must be between 0,ni-1 !
		for(int c=0; c<nc; c++) cfield[c+x*nc]=phases[c+index*p.ncomponents];
	}
	
}






void ic_phase_random(Real *cfield, params &p, const int nc, std::default_random_engine& g){
	printf("ic: %s\n",p.ic.c_str());
	
	// requires external phase-lookup file
	std::vector<Real> phases;				// initialize empty vector
	get_limitCycleData(phases,p.ncomponents,p.pthfn_lc);
	int ni=phases.size()/p.ncomponents;
	
	// init for random number generator
	if(p.spatial_noise_settings.size()==1) p.spatial_noise_settings={0,1};			// default settings, otherwise: ini settings, {0,1} = phases from {0,2pi}
	if(p.spatial_noise_settings[1]>1){ printf("Error: spatial_noise_settings in ic_phase_random must be in [0,1]\n"); exit(EXIT_FAILURE);}
	std::uniform_real_distribution<double> distribution(p.spatial_noise_settings[0],p.spatial_noise_settings[1]);
	
	// set ic
	for(int x = 0; x < p.nx; ++x){
		int index = distribution(g)*(ni-1);							// index must be between 0,ni-1 !
		for(int c=0; c<p.ncomponents; c++) cfield[c+x*nc] = phases[c+index*p.ncomponents];
	}
}






// transpose non-square array
// works from c+i*nc -> i+c*n
// backward direction will probably fail
void transpose(Real*& a, int w, int h){
	
	Real *b = new Real[w*h]{};
	
	for(int i=0; i<w; i++){
	for(int j=0; j<h; j++){
		b[i+j*w] = a[j+i*h];
	}}
	std::swap(a,b);
	
	delete[] b;
}




// fields have the form: (u1,v1, u2,v2, uN,vN); better for GPU performance!
void initialCondition(Real*& c, params &p, modelparams &m){
	
	// initialize parameters
	std::default_random_engine g(p.rngState);				// generator for random numbers
	int nc = p.ncomponents;
	
	// set different initial conditions
	if(p.ic=="phase_gaussian_envelope"){ ic_phase_gaussian_envelope(c,p,nc,g); }
	else if(p.ic=="phase_random"){ ic_phase_random(c,p,nc,g); }
	else{ std::cout << "Error: ic \"" << p.ic << "\" does not exist!" << std::endl; exit(EXIT_FAILURE); }
	
	// change structure of data from concentration-major to cell-major if CPU (different data layout than GPU)
	if(p.CPUQ) transpose(c,p.n,p.ncomponents);
	
	// save rng state for later to seed generator via generator.seed(p.rngState)
	std::stringstream rngStateStream;
	rngStateStream << g;
	rngStateStream >> p.rngState;
}

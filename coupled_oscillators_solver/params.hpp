#ifndef __params_hpp__
#define __params_hpp__

#ifdef DOUBLE
	typedef double Real;
#else
	typedef float Real;
#endif

// default values in ini / cmdline
// static global declaration here: all members are init with 0 (int, float, double) or NULL (char)
struct params{
	
	// [numerics_space]
	int nx, couplingChoice;
	Real dx;
	std::string bc;
	
	// [numerics_time]
	int nStatesSaveBuffer, saveSingleQ, timeEvolutionChoice;
	Real dt;
	size_t stepsSaveState, stepsSaveStateOffset, stepsEnd;
	float delayTimeMax, delayStartTime, delayTimeMin, delayTimeIntervalUntilMaxReached, couplingStartTime;
	
	// [device]
	int blockWidth, use_tiles, nCpuCoreThresh;
	
	// [dir_management]
	std::string pthout;
	
	// [initial_condition]
	std::string ic, pthfn_lc;
	int uSeed;
	std::string spatial_settings1_string, spatial_noise_settings_string;
	
	// [model_parameters]
	int reactionModel, hehoflag, pSeed;
	Real het1;
	float pSigma;
	std::string coupling_coeffs_string, phases_string, modelParams_string;
	
	// [nonlocal]
	Real K, kappa, kernelWidth;
	int cutoffRange;
	
	// [network]
	int network_type, networkQ;
	
	// [non-ini]
	int saveQ, n, spaceDim, ncomponents, delayFlag, kradius, kdia, CPUQ, seed;
	unsigned int rngState;
	size_t stepsCouplingStart, delayStartSteps, delayStepsMin, delayStepsMax;
	Real ksum;
	std::string pthexe;
	std::vector<Real> coupling_coeffs, spatial_settings1, spatial_noise_settings;
	
};

struct modelparams{
	int ncomponents;
	int nparams;
	Real *model_params;
	Real *model_phases;				// excited(ncomponents), refractory(ncomponents), background(ncomponents) for each component; 
	Real *coupling_coeffs;			// nonlocal coupling or diffusion
};

void translateArrayOrder(Real *c, Real *cAnalysis, params &p, int &untranslatedQ, int analysisArrayQ);









#endif	// __params_hpp__

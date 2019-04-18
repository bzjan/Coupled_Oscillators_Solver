

// C++11 & Boost libraries, OpenMP
#include <boost/filesystem.hpp>					// create_directory(), create_directories(), fs::path, fs::is_directory(), fs::remove_all(), fs::is_regular_file(), fs::copy_file(), fs::directory_iterator()
#include <boost/program_options.hpp>			// po::options_description, po::variables_map, ...
#include <boost/property_tree/ptree.hpp>		// pt::ptree
#include <boost/property_tree/ini_parser.hpp>	// pt::ini_parser::read_ini()
#include <iostream>								// std::cout, std::endl, std::cerr

#ifndef GPU
#include <omp.h>								// omp_get_max_threads(), omp_set_num_threads()
#endif

#ifdef __APPLE__				// MAC
#include <mach-o/dyld.h>
#elif _WIN32				// Win32 or Win64
#include <Windows.h>
#endif

#include "../params.hpp"					// structs params, modelparams

// namespaces
namespace fs = boost::filesystem;
namespace po = boost::program_options;
namespace pt = boost::property_tree;



std::string get_executable_path(){
	
	// initialize
	std::string pthexe;
	char buffer[1024];
	
	#ifdef __linux__			// Ubuntu
		ssize_t len = ::readlink("/proc/self/exe", buffer, sizeof(buffer)-1);
		
		// get pthfnexe
		if(len != -1){				// success
			buffer[len] = '\0';
			pthexe = std::string(buffer);
		}else{							// error
			printf("Error! Could not find path to executable!\n"); exit(EXIT_FAILURE);
		}
	
	#elif __APPLE__				// MAC
		uint32_t size = sizeof(buffer);
		if( _NSGetExecutablePath(buffer, &size) == 0 ){				// success
			pthexe = std::string(buffer);
		}else{														// error
			printf("Error: pthexe buffer too small!\n"); exit(EXIT_FAILURE);
		}
	
	#elif _WIN32				// Win32 or Win64
		if( GetModuleFileName(NULL, buffer, MAX_PATH) ){			// success
			pthexe = std::string(buffer);
		}else{														// error
			printf("Error reading pthexe on windows!\n"); exit(EXIT_FAILURE);
		}
	
	#else
		printf("Error: OS not recognized!\n"); exit(EXIT_FAILURE);
	
	#endif
	
	// remove filename from pathexe
	const size_t last_slash_idx = pthexe.rfind('/');
	if (std::string::npos != last_slash_idx){
		pthexe = pthexe.substr(0, last_slash_idx+1);
	}
	
	return pthexe;
}



// custom parser for arrays in inifile, works for arrays of any kind: string, int, float, double...
template<typename T>
std::vector<T> to_array(const std::string& s){
	std::vector<T> result;
	std::stringstream ss(s);
	std::string item;
	while(std::getline(ss, item, ',')) result.push_back(boost::lexical_cast<T>(item));
	return result;
}


std::string RealArrayToString(Real *array, const int n){
	
	std::string result="";
	std::ostringstream strs;
	
	for(int i=0; i<n-1; i++) strs << array[i] << ",";
	strs << array[n-1];
	
	return strs.str();
}


std::string RealVectorToString(std::vector<Real> vec){
	
	std::string result="";
	std::ostringstream strs;
	
	for(int i=0; i<(int)vec.size()-1; i++) strs << vec[i] << ",";
	strs << vec[vec.size()-1];
	
	return strs.str();
}


std::string replaceEnvironmentVariables(std::string input){
	
	size_t pos1 = input.find("$", 0);
	if(pos1 != std::string::npos){										// "$" was found in string
		size_t pos2 = input.find("/", pos1);
		std::string envVarString = input.substr(pos1+1,pos1+pos2-1).c_str();
		std::string envVarValue;
		if( getenv(envVarString.c_str()) != NULL ) envVarValue = getenv(envVarString.c_str());
		return input.replace(pos1,pos1+pos2,envVarValue);
	}else{
		return input;
	}
}


// read command line parameters with boost
void read_cmdline(int &ac, char *av[], std::string &pthini, std::string &ini, params &p, po::variables_map &vm){
	
	int errorQ=0;			// no error: 0, error: 1
	
	try{
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "produce help message")
			
			// cmdline options to specify inifile location
			("pthini", po::value<std::string>(&pthini)->default_value(p.pthexe+"../ini"), "path to inifile")
			("ini", po::value<std::string>(&ini)->default_value("default2d.ini"), "name of inifile")
			
			// [numerics_space]
			("numerics_space.nx", po::value<int>(), "nx")
			("numerics_space.dx", po::value<Real>(), "dx")
			("numerics_space.couplingChoice", po::value<int>(), "couplingChoice")
			
			// [numerics_time]
			("numerics_time.stepsEnd", po::value<size_t>(), "stepsEnd")
			("numerics_time.stepsSaveState", po::value<size_t>(), "stepsSaveState")
			("numerics_time.delayTimeMax", po::value<float>(), "delayTimeMax")
			("numerics_time.dt", po::value<Real>(), "dt")
			
			// [device]
			("device.gpu_blockWidth", po::value<int>(), "blockWidth")
			("device.gpu_use_tiles", po::value<int>(), "use GPU tiling algorithm?")
			
			// [initial_condition]
			("initial_condition.spatial_settings1", po::value<std::string>(), "excitable array, no whitespace!")
			("initial_condition.uSeed", po::value<int>(), "uSeed")
			
			// [model_parameters]
			("model_parameters.reactionModel", po::value<int>(), "reactionModel")
			("model_parameters.coupling_coeffs", po::value<std::string>(), "coupling/diffusion coefficents, no whitespace!")
			("model_parameters.pSeed", po::value<int>(), "seed for parameter heterogeneity")
			("model_parameters.hehoflag", po::value<int>(), "hehoflag")
			
			// [nonlocal]
			("nonlocal.cutoffRange", po::value<int>(), "cutoffRange")
			("nonlocal.K", po::value<Real>(), "K")
			("nonlocal.kappa", po::value<Real>(), "kappa")
			
			// [dir_management]
			("dir_management.pthout", po::value<std::string>(), "path for output")
			;
		
		po::store(po::parse_command_line(ac, av, desc), vm);
		po::notify(vm);
		
		if(vm.count("help")){ std::cout << desc << std::endl; exit(EXIT_SUCCESS); }
		
	}
	catch(std::exception& e){
		std::cerr << "error: " << e.what() << "\n";
		errorQ=1;
	}
	catch(...){
		std::cerr << "Exception of unknown type!\n";
		errorQ=1;
	}
	
	if(errorQ){
		std::cout<<"--- program shutdown due to error in function read_cmdline() ---"<<std::endl; 
		exit(EXIT_FAILURE);
	}
	
}



// strip string of unwanted chars
std::string stripString(std::string &str){
	
	// escaped quotation mark
	char badchars[] = "\"";
	
	// you need include <algorithm> to use general algorithms like std::remove()
	for(unsigned int i = 0; i < strlen(badchars); ++i) str.erase (std::remove(str.begin(), str.end(), badchars[i]), str.end());
	return str;
}


void get_defaultReactionCouplingParams(params &p, modelparams &m){
	// phases = { excited(n) }, { refractory(n) }, { background(n) }
	
	// switch with module "{" "}" would work!
	switch(p.reactionModel){
			
		case 16:												// phase model
		case 1601:												// phase model_whet
			m.ncomponents=1;
			m.nparams=1;
			m.model_params = new Real[1] {1.0};					// w
			m.model_phases = new Real[3] {4.2,2.1,0.0};			// exc, inh, background
			m.coupling_coeffs = new Real[1] {0.0};				// no diffusion coefficient, but alpha in nonlocal coupling
			break;
			
			
		case 24:												// zbke2k
		case 2401:												// zbke2k qhet
			m.ncomponents=2;
			m.nparams=9;
			m.model_params = new Real[9] {1.0/0.11,1.7e-5,1.6e-3,0.1,1.7e-5,1.2,2.4e-4,0.7,5.25e-4};		// ooeps1, eps2, eps3, alpha, beta, gamma, mu, q, phi0
			m.model_phases = new Real[6] {0.1,0.01,0.0001,0.3,1e-4,1e-4};
			m.coupling_coeffs = new Real[2] {1.0/0.11,2.0};
			break;
			
			
		default:												// default case
			printf("Error: reactionModel not implemented. (readInifile @ line: %d)\n",__LINE__); exit(EXIT_FAILURE);
			break;
	}
}


void set_ReactionCouplingParams(params &p, pt::ptree pt, modelparams &m){
	
	get_defaultReactionCouplingParams(p,m);
	
	// translate to vectors
	std::vector<Real> modelParameters0 = to_array<Real>(pt.get<std::string>("model_parameters.modelParams","0"));
	std::vector<Real> coupling_coeffs0 = to_array<Real>(p.coupling_coeffs_string);
	std::vector<Real> phases0 = to_array<Real>(pt.get<std::string>("model_parameters.phases","0"));
	
	
	// use default values if not set by ini or cmdline; then write values into resulting ini file
	if(modelParameters0.size()!=1 or modelParameters0[0]!=0){
		for(size_t i=0; i<modelParameters0.size(); i++) m.model_params[i]=modelParameters0[i];
	}else{
		std::string s = RealArrayToString(m.model_params,m.nparams);
		pt.put<std::string>("model_parameters.modelParams",s);
	}
	
	if(coupling_coeffs0.size()!=1 or coupling_coeffs0[0]!=0){
		for(size_t i=0; i<coupling_coeffs0.size(); i++) m.coupling_coeffs[i]=coupling_coeffs0[i];
	}else{
		std::string s=RealArrayToString(m.coupling_coeffs,m.ncomponents);
		pt.put<std::string>("model_parameters.coupling_coeffs",s);
	}
	
	if(phases0.size()!=1 or phases0[0]!=0){
		for(size_t i=0; i<phases0.size(); i++) m.model_phases[i]=phases0[i];
	}else{
		std::string s = RealArrayToString(m.model_phases,m.ncomponents*3);
		pt.put<std::string>("model_parameters.phases",s);
	}
	
}



// translate po::variables_map to pt::ptree
void translate_variables_map_to_ptree(po::variables_map &vm, pt::ptree &propTree){
	
	for(po::variables_map::iterator it=vm.begin(); it!=vm.end(); it++){
		if( it->second.value().type() == typeid(int) ){ propTree.put<int>(it->first,vm[it->first].as<int>()); }
		else if( it->second.value().type() == typeid(float) ){ propTree.put<float>(it->first,vm[it->first].as<float>()); }
		else if( it->second.value().type() == typeid(double) ){ propTree.put<double>(it->first,vm[it->first].as<double>()); }
		else if( it->second.value().type() == typeid(std::string) ){ propTree.put<std::string>(it->first,vm[it->first].as<std::string>()); }
		else if( it->second.value().type() == typeid(size_t) ){ propTree.put<size_t>(it->first,vm[it->first].as<size_t>()); }
		else{ printf("Error: unknown datatype for cmdline option. Abort!\n"); exit(EXIT_FAILURE); }
	}
}



// read parameters from text file
void read_inifile(const std::string pthfnini, params &p, pt::ptree &propTree){
	
	// read file for proptree
	pt::ini_parser::read_ini(pthfnini, propTree);
	
	// read inifile for program_options
	int errorQ=0;			// no error: 0, error: 1
	std::ifstream pthfnini_stream(pthfnini);
	
	std::cout<<"\n############# reading ini file ###################"<<std::endl;
	std::cout<<pthfnini<<std::endl;
	try{
		po::options_description inifile_options("Allowed inifile options");
		inifile_options.add_options()
		
		// [numerics_space]
		("numerics_space.nx", po::value<int>(&p.nx)->default_value(1), "number of cells in x direction")
		("numerics_space.dx", po::value<Real>(&p.dx)->default_value(1.0), "resolution x direction")
		("numerics_space.boundary_condition", po::value<std::string>(&p.bc)->default_value("periodic"), "type of boundary condition")				// periodic or neumann
		("numerics_space.couplingChoice", po::value<int>(&p.couplingChoice)->default_value(0), "type of coupling")
		
		// [numerics_time]
		("numerics_time.dt", po::value<Real>(&p.dt)->default_value(0.01), "time resolution")
		("numerics_time.stepsSaveState", po::value<size_t>(&p.stepsSaveState)->default_value(10000), "save data every nth step")
		("numerics_time.stepsSaveStateOffset", po::value<size_t>(&p.stepsSaveStateOffset)->default_value(0), "start at offset to save data")
		("numerics_time.nStatesSaveBuffer", po::value<int>(&p.nStatesSaveBuffer)->default_value(100), "number of states to be held in buffer")
		("numerics_time.stepsEnd", po::value<size_t>(&p.stepsEnd)->default_value(100000), "total number of steps in simulation")
		("numerics_time.couplingStartTime", po::value<float>(&p.couplingStartTime)->default_value(0.0), "time at which coupling is activated")
		("numerics_time.delayTimeMax", po::value<float>(&p.delayTimeMax)->default_value(0.0), "max of changing time delay")
		("numerics_time.delayTimeMin", po::value<float>(&p.delayTimeMin)->default_value(0.0), "min of changing time delay")
		("numerics_time.delayTimeIntervalUntilMaxReached", po::value<float>(&p.delayTimeIntervalUntilMaxReached)->default_value(0.0), "time until delay time max is reached")
		("numerics_time.delayStartTime", po::value<float>(&p.delayStartTime)->default_value(1), "time at which time delay coupling starts")
		("numerics_time.timeEvolutionChoice", po::value<int>(&p.timeEvolutionChoice)->default_value(0), "time evolution scheme, 0 - euler (default)")
		("numerics_time.saveSingleQ", po::value<int>(&p.saveSingleQ)->default_value(0), "save data on each step (useful for debug)")
		
		// [device]
		("device.gpu_blockWidth", po::value<int>(&p.blockWidth)->default_value(8), "size of blocks on gpu in one direction")
		("device.gpu_use_tiles", po::value<int>(&p.use_tiles)->default_value(0), "use tiling algorithm?")
		("device.cpu_nCoreThresh", po::value<int>(&p.nCpuCoreThresh)->default_value(100), "threshold of sites at which to increase number of cores by one")
		
		// [initial_condition]
		("initial_condition.ic", po::value<std::string>(&p.ic)->default_value("uniform_noise"), "type of initial condition")
		("initial_condition.pthfn_lc", po::value<std::string>(&p.pthfn_lc)->default_value(p.pthexe+"/../limit_cycle_data/zbke_0_prl13_1period.dat"), "path to limit cycle data used for phase initial condition")
		("initial_condition.uSeed", po::value<int>(&p.uSeed)->default_value(0), "random seed for concentration fields")
		("initial_condition.spatial_settings1", po::value<std::string>(&p.spatial_settings1_string)->default_value("0"), "spatial settings options 1")
		("initial_condition.spatial_noise_settings", po::value<std::string>(&p.spatial_noise_settings_string)->default_value("0"), "concentration noise settings")
		
		// [model_parameters]
		("model_parameters.reactionModel", po::value<int>(&p.reactionModel)->default_value(10), "choose reaction model")
		("model_parameters.hehoflag", po::value<int>(&p.hehoflag)->default_value(0), "type of parameter heterogeneity")
		("model_parameters.pSeed", po::value<int>(&p.pSeed)->default_value(0), "random seed for random parameter distribution")
		("model_parameters.pSigma", po::value<float>(&p.pSigma)->default_value(0.01), "width of parameter distribution")
		("model_parameters.het1", po::value<Real>(&p.het1)->default_value(0.01), "heterogeneity parameter 1")
		("model_parameters.coupling_coeffs", po::value<std::string>(&p.coupling_coeffs_string)->default_value("0"), "species-wise coupling coefficients")
		("model_parameters.phases", po::value<std::string>(&p.phases_string)->default_value("0"), "values for phases (exc,ref,bg)")
		("model_parameters.modelParams", po::value<std::string>(&p.modelParams_string)->default_value("0"), "logged model parameters, not used")
		
		// [nonlocal]
		("nonlocal.K", po::value<Real>(&p.K)->default_value(1.0), "nonlocal coupling strength")
		("nonlocal.kappa", po::value<Real>(&p.kappa)->default_value(3.0), "characteristic coupling range")
		("nonlocal.cutoffRange", po::value<int>(&p.cutoffRange)->default_value(3), "maximum width of kertnel in one direction")
		
		// [network]
		("network.networkQ", po::value<int>(&p.networkQ)->default_value(0), "use network coupling? 0 - false, 1 - true")
		
		// [dir_management]
		("dir_management.pthout", po::value<std::string>(&p.pthout)->default_value(p.pthexe+"/../../Simulations/test"), "path for writing output")
		;
		
		po::variables_map vm;
		po::store(po::parse_config_file(pthfnini_stream, inifile_options), vm);
		po::notify(vm);
	}
	catch(std::exception& e){
		std::cerr << "error: " << e.what() << "\n";
		errorQ=1;
	}
	catch(...){
		std::cerr << "Exception of unknown type!\n";
		errorQ=1;
	}
	
	pthfnini_stream.close();
	if(errorQ){ std::cout<<"--- program shutdown due to error in read_inifile ---"<<std::endl; exit(EXIT_FAILURE); }
}



void update_params_from_cmdline(pt::ptree &propTree, po::variables_map &vm, params &p){
	
	// initialize options
	std::string opsName;
	
	// data: vmap from cmdline to propTree from inifile
	translate_variables_map_to_ptree(vm,propTree);
	
	// assign/overwrite cmd line values to params
	// [numerics_space]
	opsName = "numerics_space.couplingChoice"; if(vm.count(opsName)) p.couplingChoice = vm[opsName].as<int>();
	opsName = "numerics_space.nx"; if(vm.count(opsName)) p.nx = vm[opsName].as<int>();
	opsName = "numerics_space.dx"; if(vm.count(opsName)) p.dx = vm[opsName].as<Real>();
	
	// [numerics_time]
	opsName = "numerics_time.dt"; if(vm.count(opsName)) p.dt = vm[opsName].as<Real>();
	opsName = "numerics_time.stepsSaveState"; if(vm.count(opsName)) p.stepsSaveState = vm[opsName].as<size_t>();
	opsName = "numerics_time.delayTimeMax"; if(vm.count(opsName)) p.delayTimeMax = vm[opsName].as<float>();
	opsName = "numerics_time.stepsEnd"; if(vm.count(opsName)) p.stepsEnd = vm[opsName].as<int>();
	
	// [device]
	opsName = "device.gpu_blockWidth"; if(vm.count(opsName)) p.blockWidth = vm[opsName].as<int>();
	opsName = "device.gpu_use_tiles"; if(vm.count(opsName)) p.use_tiles = vm[opsName].as<int>();
	
	// [initial_condition]
	opsName = "initial_condition.uSeed"; if(vm.count(opsName)) p.uSeed = vm[opsName].as<int>();
	opsName = "initial_condition.spatial_settings1"; if(vm.count(opsName)) p.spatial_settings1_string = vm[opsName].as<std::string>();		// "_string" suffix is removed for shorter option
	
	// [model_parameters]
	opsName = "model_parameters.coupling_coeffs"; if(vm.count(opsName)) p.coupling_coeffs_string = vm[opsName].as<std::string>();
	opsName = "model_parameters.reactionModel"; if(vm.count(opsName)) p.reactionModel = vm[opsName].as<int>();
	opsName = "model_parameters.pSeed"; if(vm.count(opsName)) p.pSeed = vm[opsName].as<int>();
	opsName = "model_parameters.hehoflag"; if(vm.count(opsName)) p.hehoflag = vm[opsName].as<int>();
	
	// [nonlocal]
	opsName = "nonlocal.cutoffRange"; if(vm.count(opsName)) p.cutoffRange = vm[opsName].as<int>();
	opsName = "nonlocal.kappa"; if(vm.count(opsName)) p.kappa = vm[opsName].as<Real>();
	opsName = "nonlocal.K"; if(vm.count(opsName)) p.K = vm[opsName].as<Real>();
	
	// [dir_management]
	opsName = "dir_management.pthout"; if(vm.count(opsName)) p.pthout = vm[opsName].as<std::string>();
	
}

// adapt number of processors based on system size
void set_OpenMP_Cores(params &p){
	
	#ifndef GPU
		int maxCPUs = omp_get_max_threads();
		int nOMPs = std::min(p.n/p.nCpuCoreThresh+1,maxCPUs);
		omp_set_num_threads(nOMPs);
		printf("# of used processors: %d/%d\n",nOMPs,maxCPUs);
	#endif
	
}


// TODO: recursive copy
// create/clean directories for output
void housekeeper(const std::string pthoutString, const std::string pthexeString, const int CPUQ){
	
	bool successQ;
	fs::path pthout(pthoutString);
	fs::path pthexe(pthexeString);
	
	// removes any and all files in output directories
	if(fs::is_directory(pthout)){
		successQ = fs::remove_all(pthout);
		if(!successQ){ std::cout << "Error: Failed removing output directory." << std::endl; exit(EXIT_FAILURE); }
	}
	
	// create directories and copy current source code into output directory
	fs::create_directories(pthout);						// create_directories works recursively
	fs::create_directory(pthout / "states");
	fs::create_directory(pthout / "source_code");
	
	for(fs::path const &f : fs::directory_iterator(pthexe)){
		if(fs::is_regular_file(f) and (f.extension()==".cpp" or f.extension()==".hpp" or f.extension()==".cu" or f.extension()==".hu" or f.filename()=="makefile")){
			fs::copy_file(f,pthout / "source_code" / f.filename());								// copy file from source to output directory
		}
	}
	
	fs::create_directory(pthout / "source_code" / "Utilities");
	for(fs::path const &f : fs::directory_iterator(pthexe / "Utilities")){
		if(fs::is_regular_file(f) and (f.extension()==".cpp" or f.extension()==".hpp" or f.extension()==".cu" or f.extension()==".hu" or f.filename()=="makefile")){
			fs::copy_file(f,pthout / "source_code" / "Utilities" / f.filename());								// copy file from source to output directory
		}
	}
	
	std::string hardwareType;
	if(CPUQ) hardwareType = "CPU";
	else hardwareType = "GPU";
	fs::create_directory(pthout / "source_code" / hardwareType);
	for(fs::path const &f : fs::directory_iterator(pthexe / hardwareType)){
		if(fs::is_regular_file(f) and (f.extension()==".cpp" or f.extension()==".hpp" or f.extension()==".cu" or f.extension()==".hu" or f.filename()=="makefile")){
			fs::copy_file(f,pthout / "source_code" / hardwareType / f.filename());								// copy file from source to output directory
		}
	}
}




// read parameters from text file
void organize_ini_options(int &argumentsCount, char *argumentsValues[], params &p, modelparams &m){
	
	// initialize variables
	p.pthexe = get_executable_path();
	std::string pthini, ini;
	pt::ptree propTree;
	po::variables_map vm;
	
	// 1. get options from cmdline; get path (pthini) and filename (ini) of inifile
	read_cmdline(argumentsCount,argumentsValues,pthini,ini,p,vm);
	const std::string pthfnini = pthini+"/"+ini;
	
	// 2. get options from inifile
	read_inifile(pthfnini,p,propTree);
	
	// 3. overwrite params from file with those from commandline
	update_params_from_cmdline(propTree,vm,p);
	
	// 4. adapt options and input checks
	// array_string to array
	p.coupling_coeffs = to_array<Real>(p.coupling_coeffs_string);
	p.spatial_settings1 = to_array<Real>(p.spatial_settings1_string);
	p.spatial_noise_settings = to_array<Real>(p.spatial_noise_settings_string);
	
	// interpret environment variables in pthout string
	p.pthout = replaceEnvironmentVariables(p.pthout);
	p.pthfn_lc = replaceEnvironmentVariables(p.pthfn_lc);
	
	// check correctness of delay settings, if inappropriate switch to easiest case
	if(p.couplingChoice==6 or p.couplingChoice==1004){ 
		p.delayFlag=1;
	}else{ 
		p.delayFlag=0; 
	}
	
	if(p.delayFlag==1 and p.delayTimeMax==0.0){
		if(p.networkQ){ 
			printf("\nProblem: delayTimeMax of 0.0 is not supported for couplingChoice=1004! \nSolution: Switching to couplingChoice=1002 (readInifile @ line: %d)\n",__LINE__);
			p.couplingChoice=1002; 
		}else{ 
			printf("\nProblem: delayTimeMax of 0.0 is not supported for couplingChoice=6! \nSolution: Switching to couplingChoice=5 (readInifile @ line: %d)\n",__LINE__);
			p.couplingChoice=5; 
		}
		p.delayFlag=0;
	}
	propTree.put<int>("numerics_space.couplingChoice",p.couplingChoice);
	
	// input checks
	if(p.delayFlag){
		if(p.delayTimeIntervalUntilMaxReached!=0){
			if(p.delayTimeMax<p.delayTimeMin){
				std::cout << "\nProblem: delayTimeMin > delayTimeMax! Correcting...\n" << std::endl; 
				p.delayTimeMax=p.delayTimeMin;
			}
			if(p.delayStartTime<p.delayTimeMin){
				std::cout << "\nProblem: delayTimeMin must be >= delayStartTime! \nSolution: Setting delayStartTime = delayTimeMin! \n" << std::endl;
				p.delayStartTime=p.delayTimeMin;
			}
		}else if(p.delayTimeIntervalUntilMaxReached==0){
			if(p.delayStartTime<p.delayTimeMax and p.couplingStartTime<p.delayTimeMax){
				std::cout << "\nProblem: delayTimeMax must be <= delayStartTime or couplingStartTime! \nSolution: Setting delayStartTime = delayTimeMax! \n" << std::endl;
				p.delayStartTime = p.delayTimeMax;
			}
		}
	}
	propTree.put<float>("numerics_time.delayStartTime",p.delayStartTime);
	propTree.put<float>("numerics_time.delayTimeMax",p.delayTimeMax);
	
	// find the space dimension automatically
	p.spaceDim=1;
	if(p.networkQ) p.spaceDim=1000;
	
	set_ReactionCouplingParams(p,propTree,m);
	p.ncomponents=m.ncomponents;
	
	
	#ifdef GPU
		p.CPUQ = 0;
	#else
		p.CPUQ = 1;
	#endif
	
	// rng initialization
	p.rngState = p.uSeed;		// ic for concentrations use rng first
	
	// time and steps conversion
	p.delayStepsMin=(p.delayTimeMin/p.dt)*!!p.delayFlag;
	p.delayStepsMax=(p.delayTimeMax/p.dt)*!!p.delayFlag;
	p.delayStartSteps=(p.delayStartTime/p.dt)*!!p.delayFlag;
	p.stepsCouplingStart=p.couplingStartTime/p.dt;
	propTree.put<size_t>("numerics_time.delayStepsMin",p.delayStepsMin);
	propTree.put<size_t>("numerics_time.delayStepsMax",p.delayStepsMax);
	propTree.put<size_t>("numerics_time.delayStartSteps",p.delayStartSteps);
	
	// set paths
	if(p.CPUQ) p.pthout=stripString(p.pthout)+"_cpu";
	else p.pthout=stripString(p.pthout)+"_gpu";
	propTree.put<std::string>("dir_management.pthout",p.pthout);
	p.pthfn_lc=stripString(p.pthfn_lc);
	
	p.bc=stripString(p.bc);
	p.ic=stripString(p.ic);
	if(p.networkQ){
		p.n=p.nx;
		printf("spaceDim: %d (network)\n",p.spaceDim);
		printf("n: %d\n",p.n);
	}else{
		p.n=p.nx;											// total number of cells
		printf("spaceDim: %d\n",p.spaceDim);
		printf("nx: %d\n",p.nx);
		printf("dx: %.2f\n",p.dx);
	}
	propTree.put<int>("numerics_space.n",p.n);
	
	// recalculate total simulation time, so that no superfluous, unsaved simulation steps are performed
	p.stepsEnd = (p.stepsEnd/p.stepsSaveState)*p.stepsSaveState;
	propTree.put<size_t>("numerics_time.stepsEnd",p.stepsEnd);
	
	
	// set number of parallel processors
	set_OpenMP_Cores(p);
	
	// create output directories
	housekeeper(p.pthout,p.pthexe,p.CPUQ);
	
	// write output ini with used options
	std::string iniOutString=p.pthout+"/used_options.ini";
	pt::ini_parser::write_ini(iniOutString,propTree);
	
	
	
	// terminal output
	#ifdef DOUBLE
		std::cout << "dataytpe mode: DOUBLE" << std::endl;
	#else
		std::cout << "dataytpe mode: FLOAT" << std::endl;
	#endif
	std::cout<<"dt: "<<p.dt<<std::endl;
	std::cout<<"reaction model: "<<p.reactionModel<<std::endl;
	std::cout<<"stepsSaveState: "<<p.stepsSaveState<<" steps = "<<p.stepsSaveState*p.dt<<" time units"<<std::endl;
	std::cout<<"stepsEnd: "<<p.stepsEnd<<" steps = "<<p.stepsEnd*p.dt<<" time units"<<std::endl;
	std::cout<<"pthout: "<<p.pthout<<std::endl;																				// DEBUG
	std::cout<<"##############################################\n"<<std::endl;
	
	
	//~ exit(EXIT_SUCCESS);		// DEBUG
}

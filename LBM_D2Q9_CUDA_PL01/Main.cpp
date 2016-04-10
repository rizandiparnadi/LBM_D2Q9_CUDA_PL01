#include <iostream>
#include <fstream>
#include "LBMSimulationGPU.cuh"
#include "Timer.h"

int main()
{
	double err = 1.0;
	int k = 0;

	LBMSimulationParameters params;
	params.Nx = 512;			// Number of horizontal lattices
	params.Ny = 512;			// Number of vertical lattices
	params.L = (params.Ny + 1);	// Length of cavity
	params.rho0 = 1.0;			// Initial density
	params.ux0 = 0.0;			// Initial x velocity
	params.uy0 = 0.0;			// Initial y velocity
	params.uw = 0.1;			// Wall velocity
	params.Re = 400.0;			// Reynolds number
	params.n = 1.5;				// Power law fluid parameter
	params.t0 = 0.36754;		// Currently unused

	LBMSimulationGPU lbm;
	lbm.initialize(params);
	lbm.initEquilibrium();

	Timer timer;
	
	while (err>1.0e-6)
	{
		k++;
		lbm.streaming();
		lbm.bounceback();
		lbm.macroscopic();
		lbm.collisionBGK();
		if (k % 100 == 0)
		{
			printf("k = %i\n", k);
		}
		if (k % 50000 == 0)
		{
			//err = lbm.error();
			//printf("error=%e k=%d\n", err, k);
			break;
		}
	}

	double time = timer.getSeconds();
	printf("\nFinished in %f seconds.\n", time);

	lbm.cpuReadback();
	lbm.volumetric();
	lbm.outputData();

	cudaDeviceReset();

	std::ofstream output;
	output.open("meta.txt");
	output << "Re " << params.Re << std::endl;
	output << "n " << params.n << std::endl;
	output << "Nx " << params.Nx << std::endl;
	output << "Ny " << params.Ny << std::endl;
	output << "error=" << err << " k=" << k << std::endl;
	output << "Finished in " << time << " seconds." << std::endl;
	output.close();

	system("PAUSE");

	return 0;
}


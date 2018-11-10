#pragma once

#include <vector>
#include "Array1D.h"
#include "Array2D.h"
#include "Array3D.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

typedef double Real;

// Requires changing notation from A[i][j][k] to A(i,j,k) 
typedef Array3D<Real> RealArray3D;
typedef Array2D<Real> RealArray2D;
typedef Array1D<Real> RealArray1D;
//typedef Array3D<Vector2<Real>> VectorArray3D;
//typedef Array2D<Vector2<Real>> VectorArray2D;

/*
#ifdef DEBUG
typedef std::vector<Real> RealArray1D;
typedef std::vector<std::vector<Real>> RealArray2D;
typedef std::vector<std::vector<std::vector<Real>>> RealArray3D;
#else
typedef std::vector<Real> RealArray1D;
typedef std::vector<std::vector<Real>> RealArray2D;
typedef std::vector<std::vector<std::vector<Real>>> RealArray3D;
#endif
*/

void initGPUIntArray(int** arr, int x);
void initGPUArray(Real** arr, int x);
void initGPUArray2D(Real** arr, int x, int y);
void initGPUArray3D(Real** arr, int x, int y, int z);

struct LBMSimulationParameters
{
	//int Q;	// Number of discrete velocities
	int Nx;		// Number of horizontal lattices
	int Ny;		// Number of vertical lattices
	Real L;		// Length of cavity
	Real rho0;	// Initial density
	Real ux0;	// Initial x velocity
	Real uy0;	// Initial y velocity
	Real uw;	// Wall velocity
	Real Re;	// Reynolds number
	Real n;		// Power law fluid parameter
	Real t0;	// Currently unused
};

// Lattice-Boltzmann Simulation class
class LBMSimulationGPU
{
public:
	void initialize(LBMSimulationParameters params);

public:
	void initEquilibrium();	// Initialization (equilibrium method)
	void streaming();		// Streaming
	void bounceback();		// Handle boundaries
	void macroscopic();		// Calculate macroscopic parameters
	void collisionBGK();	// Collision approximated by the Bhatnagar-Gross-Krook model

	void cpuReadback();		// Read back gpu buffers to cpu memory
	void volumetric();
	void outputData();		// Output data
	Real error();			// Error

	Real feq(Real rho, Real ux, Real uy, int k); // Equilibrium distribution function

protected:
	typename<T>
	void cpuReadbackArray(T array, Real* device_array);
	
	dim3 dimBlock, dimGrid;

	///////// GPU-side variables
	int* d_N;		// X & Y dimensions
	Real* d_f;		// Distribution function
	Real* d_f_post;	// Post streaming distribution function
	Real* d_rho;	// Density
	Real* d_ux;		// X Velocity
	Real* d_uy;		// Y Velocity
	Real* d_x;
	Real* d_y;
	Real* d_tau_ap;

	Real* d_tau;	// Relaxation time
	Real* d_rho0;	// Initial density
	Real* d_uw;		// Wall velocity

	Real* d_ux_prev;
	Real* d_uy_prev;
	Real* d_umag;	// Norms of velocity
	Real* d_u_hor;	// Horizontal velocity at the center of the cavity
	Real* d_u_ver;	// Vertical velocity at the center of the cavity

	Real* d_vol_hor;
	Real* d_vol_ver;
	Real* d_err;

	///////// CPU-side variables
	RealArray2D rho;
	RealArray2D ux;
	RealArray2D uy;
	RealArray3D f;
	RealArray2D tau_ap;
	RealArray1D u_hor;
	RealArray1D u_ver;
	RealArray1D x;
	RealArray1D y;
	Real vol_hor;
	Real vol_ver;

	Real tau;

	int Nx;		// Number of horizontal lattices
	int Ny;		// Number of vertical lattices
	Real L;		// Length of cavity
	Real rho0;	// Initial density
	Real ux0;	// Initial x velocity
	Real uy0;	// Initial y velocity
	Real uw;	// Wall velocity
	Real Re;	// Reynolds number
	Real n;		// Power law fluid parameter
	Real t0;	// Currently unused

	int iter;	// Number of iterations
};

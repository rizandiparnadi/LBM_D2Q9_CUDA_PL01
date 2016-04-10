#pragma once

#include "LBMSimulationGPU.cuh"
#include <iostream>
#include <fstream>


// In case Koplo uses shitty mac standards again
#ifdef WIN32
#define popen _popen
#define pclose _pclose
#endif

#define Q 9
__constant__ int cx[9];
__constant__ int cy[9];
__constant__ Real w[9];
__constant__ int N[2];

void initGPUIntArray(int** arr, int x)
{
	cudaError_t cudaStatus = cudaMalloc((void**)arr, sizeof(int)*x);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMalloc failed :" << cudaGetErrorName(cudaStatus) << std::endl;
}

void initGPUArray(Real** arr, int x)
{
	cudaError_t cudaStatus = cudaMalloc((void**)arr, sizeof(Real)*x);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMalloc failed :" << cudaGetErrorName(cudaStatus) << std::endl;
}

void initGPUArray2D(Real** arr, int x, int y)
{
	cudaError_t cudaStatus = cudaMalloc((void**)arr, sizeof(Real)*x*y);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMalloc failed :" << cudaGetErrorName(cudaStatus) << std::endl;
}

void initGPUArray3D(Real** arr, int x, int y, int z)
{
	cudaError_t cudaStatus = cudaMalloc((void**)arr, sizeof(Real)*x*y*z);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMalloc failed :" << cudaGetErrorName(cudaStatus) << std::endl;
}

// CUDA kernel functions
__global__ void kernelInitEquilibrium(Real* f, Real* rho, Real* ux, Real* uy, Real* rho0);
__global__ void kernelStreaming(Real* f, Real* f_post);
__global__ void kernelMacroscopic(Real* f, Real* f_post, Real* rho, Real* ux, Real* uy);
__global__ void kernelCollisionBGK(Real* f, Real* f_post, Real* rho, Real* ux, Real* uy, Real* tau_ap, Real m, Real n);
__global__ void kernelBouncebackTop(Real* f, Real* f_post, Real* rho, Real* uw);
__global__ void kernelBouncebackBottom(Real* f, Real* f_post, Real* rho);
__global__ void kernelBouncebackLeft(Real* f, Real* f_post, Real* rho);
__global__ void kernelBouncebackRight(Real* f, Real* f_post, Real* rho);

__device__ Real feq(Real rho, Real ux, Real uy, int k); // Equilibrium distribution function

void LBMSimulationGPU::initialize(LBMSimulationParameters params)
{
	Nx = params.Nx;		// Number of horizontal lattices
	Ny = params.Ny;		// Number of vertical lattices
	L = params.L;		// Length of cavity
	rho0 = params.rho0;	// Initial density
	ux0 = params.ux0;	// Initial x velocity
	uy0 = params.uy0;	// Initial y velocity
	uw = params.uw;		// Wall velocity
	Re = params.Re;		// Reynolds number
	n = params.n;		// Powe law fluid parameter
	t0 = params.t0;

	tau = 3 * L*uw / Re + 0.5;	// Relaxation time
	//tau = (6.0*v-1.0)/2.0;

	// Initialize arrays to their proper dimensions
	initGPUArray3D(&d_f, Ny, Nx, Q);
	initGPUArray3D(&d_f_post, Ny, Nx, Q);
	initGPUArray2D(&d_rho, Ny, Nx);
	initGPUArray2D(&d_ux, Ny, Nx);
	initGPUArray2D(&d_uy, Ny, Nx);
	initGPUArray2D(&d_tau_ap, Ny, Nx);
	initGPUArray(&d_x, Nx);
	initGPUArray(&d_y, Ny);
	initGPUArray2D(&d_ux_prev, Ny, Nx);
	initGPUArray2D(&d_uy_prev, Ny, Nx);
	initGPUArray(&d_tau, 1);
	initGPUArray(&d_rho0, 1);
	initGPUArray(&d_uw, 1);

	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(d_tau, &tau, sizeof(Real), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMemcpy failed :" << cudaGetErrorName(cudaStatus) << std::endl;

	cudaStatus = cudaMemcpy(d_rho0, &rho0, sizeof(Real), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMemcpy failed :" << cudaGetErrorName(cudaStatus) << std::endl;

	cudaStatus = cudaMemcpy(d_uw, &uw, sizeof(Real), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMemcpy failed :" << cudaGetErrorName(cudaStatus) << std::endl;

	int _cx[9] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
	int _cy[9] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };
	Real _w[9] = { 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0 };
	int _N[2] = { Nx, Ny };

	cudaStatus = cudaMemcpyToSymbol(cx, &_cx[0], sizeof(int) * 9);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMemcpyToSymbol failed :" << cudaGetErrorName(cudaStatus) << std::endl;

	cudaStatus = cudaMemcpyToSymbol(cy, &_cy[0], sizeof(int) * 9);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMemcpyToSymbol failed :" << cudaGetErrorName(cudaStatus) << std::endl;

	cudaStatus = cudaMemcpyToSymbol(w, &_w[0], sizeof(Real) * 9);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMemcpyToSymbol failed :" << cudaGetErrorName(cudaStatus) << std::endl;

	cudaStatus = cudaMemcpyToSymbol(N, &_N[0], sizeof(int) * 2);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMemcpyToSymbol failed :" << cudaGetErrorName(cudaStatus) << std::endl;

	// Assign a 3D distribution of CUDA "threads" within each CUDA "block"    
	int threadsAlongX = 16, threadsAlongY = 16;
	dimBlock = dim3(threadsAlongX, threadsAlongY, 1);

	// Calculate number of blocks along X and Y in a 2D CUDA "grid"
	dimGrid = dim3(ceil(float(Nx) / float(dimBlock.x)), ceil(float(Ny) / float(dimBlock.y)), 1);
}

void LBMSimulationGPU::initEquilibrium()
{
	kernelInitEquilibrium << <dimGrid, dimBlock >> >(d_f, d_rho, d_ux, d_uy, d_rho0);
}

void LBMSimulationGPU::streaming()
{
	kernelStreaming << <dimGrid, dimBlock >> >(d_f, d_f_post);
}

void LBMSimulationGPU::bounceback()
{
	kernelBouncebackTop << <1, Nx >> >(d_f, d_f_post, d_rho, d_uw);
	kernelBouncebackBottom << <1, Nx >> >(d_f, d_f_post, d_rho);
	kernelBouncebackLeft << <1, Ny >> >(d_f, d_f_post, d_rho);
	kernelBouncebackRight << <1, Ny >> >(d_f, d_f_post, d_rho);
}

void LBMSimulationGPU::macroscopic()
{
	kernelMacroscopic << < dimGrid, dimBlock >> >(d_f, d_f_post, d_rho, d_ux, d_uy);
}

void LBMSimulationGPU::collisionBGK()
{
	Real m = pow(uw, 2.0 - n)*pow(L, n) / Re;
	kernelCollisionBGK << <dimGrid, dimBlock >> >(d_f, d_f_post, d_rho, d_ux, d_uy, d_tau_ap, m, n);
}

__device__ int getIndex(int i, int j)
{
	return i + (j * N[0]);
}

__device__ int getIndexPadded(int i, int j)
{
	return i + (j * (N[0] + 2)) + 1 + (N[0] + 2);
}

__device__ int getIndexf(int i, int j, int k)
{
	//return k + (i * Q) + (j * Q * N[0]);
	return i + (j * N[0]) + (k * N[1] * N[0]);
}

__device__ int calculateThreadIndexX()
{
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int calculateThreadIndexY()
{
	return blockIdx.y * blockDim.y + threadIdx.y;
}

__device__ Real feq(Real rho, Real ux, Real uy, int k)
{
	Real U = cx[k] * ux + cy[k] * uy;
	Real V_2 = ux*ux + uy*uy;

	return w[k] * rho*(1.0 + 3.0*U + 4.5*U*U - 1.5*V_2);
}

__global__ void kernelInitEquilibrium(Real* f, Real* rho, Real* ux, Real* uy, Real* rho0)
{
	// (x,y) index
	int i = calculateThreadIndexX();
	int j = calculateThreadIndexY();

	// Unique thread index
	int index = getIndex(i, j);

	// Every lattice is initially at equilibrium
	if (index < (N[0] * N[1]))
	{
		rho[index] = *rho0;
		ux[index] = 0.0;
		uy[index] = 0.0;

#pragma unroll
		for (int k = 0; k < 9; k++)
		{
			f[getIndexf(i, j, k)] = feq(rho[index], ux[index], uy[index], k);
		}
	}
}

__global__ void kernelStreaming(Real* f, Real* f_post)
{
	// (x,y) index
	int i = calculateThreadIndexX();
	int j = calculateThreadIndexY();

	// Unique thread index
	int index = getIndex(i, j);

	if (index < (N[0] * N[1]))
	{
#pragma unroll
		for (int k = 0; k < 9; k++)
		{
			// Index of the neighbor we stream from
			int id = i - cx[k];
			int jd = j - cy[k];

			// Make sure we don't cross boundaries
			if (id >= 0 && jd >= 0 && id < N[0] && jd < N[1])
				// Stream. f_post is post-streaming distribution function
				f_post[getIndexf(i, j, k)] = f[getIndexf(id, jd, k)];
		}
	}
}

__global__ void kernelMacroscopic(Real* f, Real* f_post, Real* rho, Real* ux, Real* uy)
{
	// (x,y) index
	int i = calculateThreadIndexX();
	int j = calculateThreadIndexY();

	// Unique thread index
	int index = getIndex(i, j);

	if (index < (N[0] * N[1]))
	{
		// Calculate macroscopic parameters
		// Density calculation
		rho[index] = f_post[getIndexf(i, j, 0)] + f_post[getIndexf(i, j, 1)] + f_post[getIndexf(i, j, 2)]
					+ f_post[getIndexf(i, j, 3)] + f_post[getIndexf(i, j, 4)] + f_post[getIndexf(i, j, 5)]
					+ f_post[getIndexf(i, j, 6)] + f_post[getIndexf(i, j, 7)] + f_post[getIndexf(i, j, 8)];

		// Velocity calculation
		ux[index] = (f_post[getIndexf(i, j, 1)] + f_post[getIndexf(i, j, 5)] + f_post[getIndexf(i, j, 8)]
					- f_post[getIndexf(i, j, 3)] - f_post[getIndexf(i, j, 6)] - f_post[getIndexf(i, j, 7)]) / rho[index];

		uy[index] = (f_post[getIndexf(i, j, 5)] + f_post[getIndexf(i, j, 6)] + f_post[getIndexf(i, j, 2)]
					- f_post[getIndexf(i, j, 7)] - f_post[getIndexf(i, j, 8)] - f_post[getIndexf(i, j, 4)]) / rho[index];
	}
}

__global__ void kernelCollisionBGK(Real* f, Real* f_post, Real* rho, Real* ux, Real* uy, Real* tau_ap, Real m, Real n)
{
	// (x,y) index
	int i = calculateThreadIndexX();
	int j = calculateThreadIndexY();

	// Unique thread index
	int index = getIndex(i, j);

	if (index < (N[0] * N[1]))
	{
		/// STEP 1 : Calculate relaxation time from strain rate tensor
		Real del_ux_plus = 0.0,
			del_ux_minus = 0.0,
			del_uy_plus = 0.0,
			del_uy_minus = 0.0;

		// TODO : pad ux and uy so we don't need to branch when handling boundaries?
		if (j<N[1] - 1)
			del_ux_plus = ux[getIndex(i, j + 1)];
		if (j>0)
			del_ux_minus = ux[getIndex(i, j - 1)];
		if (i<N[0] - 1)
			del_uy_plus = uy[getIndex(i + 1, j)];
		if (i>0)
			del_uy_minus = uy[getIndex(i - 1, j)];

		// Calculate strain rate tensor with second-order finite difference
		Real Sab_xx = (del_ux_plus - 2.0*ux[index] + del_ux_minus);
		Real Sab_yy = (del_uy_plus - 2.0*uy[index] + del_uy_minus);
		Real Sab_xy = ((del_ux_plus - 2.0*ux[index] + del_ux_minus) + (del_uy_plus - 2.0*uy[index] + del_uy_minus)) / 2.0;

		// Second invariant of the strain rate tensor
		Real D2 = Sab_xx*Sab_xx + Sab_xy*Sab_xy + Sab_xy*Sab_xy + Sab_yy*Sab_yy;

		// Shear rate
		Real gamma = 2.0*sqrt(D2);

		// Local apparent kinematic viscosity
		// Real m = pow(uw, 2.0 - n)*pow(L, n) / Re;
		Real Vap = m*pow(abs(gamma), Real(n - 1.0));

		// Local apparent relaxation time
		tau_ap[index] = (6.0*Vap + 1.0) / 2.0;

		/// STEP 2 : Perform collision
#pragma unroll
		for (int k = 0; k < 9; k++)
		{
			// Get new equilibrium distribution using updated density & velocity
			Real FEQ = feq(rho[index], ux[index], uy[index], k);
			// BGK collision
			f[getIndexf(i, j, k)] = f_post[getIndexf(i, j, k)] - (f_post[getIndexf(i, j, k)] - FEQ) / tau_ap[index];
		}
	}
}

////// Bounce back for the 4 walls
__global__ void kernelBouncebackTop(Real* f, Real* f_post, Real* rho, Real* uw)
{
	// (x,y) index
	int i = calculateThreadIndexX();

	int Ny = N[1] - 1;

	// Using Ladd's bounce scheme for moving walls
	f_post[getIndexf(i, Ny, 4)] = f[getIndexf(i, Ny, 2)];
	f_post[getIndexf(i, Ny, 7)] = f[getIndexf(i, Ny, 5)] + 6.0*rho[getIndex(i, Ny)] * w[7] * cx[7] * (*uw);
	f_post[getIndexf(i, Ny, 8)] = f[getIndexf(i, Ny, 6)] + 6.0*rho[getIndex(i, Ny)] * w[8] * cx[8] * (*uw);
}

__global__ void kernelBouncebackBottom(Real* f, Real* f_post, Real* rho)
{
	// (x,y) index
	int i = calculateThreadIndexX();

	f_post[getIndexf(i, 0, 2)] = f[getIndexf(i, 0, 4)];
	f_post[getIndexf(i, 0, 5)] = f[getIndexf(i, 0, 7)];
	f_post[getIndexf(i, 0, 6)] = f[getIndexf(i, 0, 8)];
}

__global__ void kernelBouncebackLeft(Real* f, Real* f_post, Real* rho)
{
	// (x,y) index
	int j = calculateThreadIndexX();

	f_post[getIndexf(0, j, 1)] = f[getIndexf(0, j, 3)];
	f_post[getIndexf(0, j, 5)] = f[getIndexf(0, j, 7)];
	f_post[getIndexf(0, j, 8)] = f[getIndexf(0, j, 6)];
}

__global__ void kernelBouncebackRight(Real* f, Real* f_post, Real* rho)
{
	// (x,y) index
	int j = calculateThreadIndexX();

	int Nx = N[0] - 1;

	f_post[getIndexf(Nx, j, 3)] = f[getIndexf(Nx, j, 1)];
	f_post[getIndexf(Nx, j, 7)] = f[getIndexf(Nx, j, 5)];
	f_post[getIndexf(Nx, j, 6)] = f[getIndexf(Nx, j, 8)];
}

void LBMSimulationGPU::cpuReadback()
{
	// Read back to CPU memory
	rho.init(Nx, Ny);
	ux.init(Nx, Ny);
	uy.init(Nx, Ny);
	f.init(Nx, Ny, Q);
	tau_ap.init(Nx, Ny);

	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(rho.data(), d_rho, rho.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMemcpy failed :" << cudaGetErrorName(cudaStatus) << std::endl;

	cudaStatus = cudaMemcpy(ux.data(), d_ux, rho.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMemcpy failed :" << cudaGetErrorName(cudaStatus) << std::endl;

	cudaStatus = cudaMemcpy(uy.data(), d_uy, rho.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMemcpy failed :" << cudaGetErrorName(cudaStatus) << std::endl;

	cudaStatus = cudaMemcpy(f.data(), d_f_post, f.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMemcpy failed :" << cudaGetErrorName(cudaStatus) << std::endl;

	cudaStatus = cudaMemcpy(tau_ap.data(), d_tau_ap, tau_ap.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		std::cout << "cudaMemcpy failed :" << cudaGetErrorName(cudaStatus) << std::endl;
}

void LBMSimulationGPU::volumetric()
{
	u_hor.init(Nx);
	u_ver.init(Ny);
	x.init(Nx);
	y.init(Ny);

	int M2 = Ny / 2, 
		N2 = Nx / 2;

	for (int i = 0; i<Nx; i++)
	{
		u_hor(i) = uy(i,M2);
		x(i) = (i + 0.5) / L;
	}

	for (int j = 0; j<Ny; j++)
	{
		u_ver(j) = ux(N2,j);
		y(j) = (j + 0.5) / L;
	}

	vol_hor = 0.0;
	vol_ver = 0.0;

	for (int i = 0; i<Nx; i++)
	{
		vol_hor += u_hor(i) / (257.0);
	}

	for (int j = 0; j<Ny; j++)
	{
		vol_ver += u_ver(j) / (257.0);
	}
}

void LBMSimulationGPU::outputData()
{
	// Output to file
	std::ofstream output;
	output.open("magnitude_gpu.csv");

	// Output velocity vector magnitude
	Real umag;
	for (int j = 0; j < Ny; j++)
	{
		for (int i = 0; i < Nx; i++)
		{
			// Calculate velocity vector magnitude
			umag = sqrt(ux(i, j) * ux(i, j) + uy(i, j) * uy(i, j));

			// Print out
			output << umag << ";";
		}
		output << std::endl;
	}
	output.close();

	// Output velocity
	output.open("ux_gpu.csv");
	for (int j = 0; j < Ny; j++)
	{
		for (int i = 0; i < Nx; i++)
		{
			output << ux(i, j) << ";";
		}
		output << std::endl;
	}
	output.close();
	output.open("uy_gpu.csv");
	for (int j = 0; j < Ny; j++)
	{
		for (int i = 0; i < Nx; i++)
		{
			output << uy(i, j) << ";";
		}
		output << std::endl;
	}
	output.close();

	// Output distribution functions
	for (int k = 0; k < Q; k++)
	{
		char str[20];
		sprintf(str, "gpu_f%i.csv", k);
		output.open(str);

		for (int j = 0; j < Ny; j++)
		{
			for (int i = 0; i < Nx; i++)
			{
				output << f(i, j, k) << ";";
			}
			output << std::endl;
		}

		output.close();
	}

	// Output apparent relaxation time
	output.open("tau_ap_gpu.csv");
	for (int j = 0; j < Ny; j++)
	{
		for (int i = 0; i < Nx; i++)
		{
			output << tau_ap(i, j) << ";";
		}
		output << std::endl;
	}
	output.close();

	// Output horizontal and vertical slices of velocity
	output.open("u_hor.csv");
	for (int i = 0; i<Nx; i++)
	{
		output << u_hor(i) << ";";
	}
	output.close();

	output.open("u_ver.csv");
	for (int i = 0; i<Ny; i++)
	{
		output << u_ver(i) << ";";
	}
	output.close();
}
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>

__global__ void cavity_flow(double* u, double* v, double* p, double* b, double* un, double* vn, double* pn, double dx, double dy, double dt, double rho, double nu, int nit, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > n*n)
		return;
	int position = 0; //inside
	if (i < n)
		position = 1; //bottom
	else if (i >= (n - 1) * n)
		position = 2; //top
	else if (i % n == 0)
		position = 3; //left
	else if ((i + 1) % n == 0)
		position = 4; //right
	//inside => build_up_b
	if (position == 0)
	{
		b[i] = rho * (1. / dt * ((u[i] - u[i - 1]) /
			(2. * dx) + (v[i + n] - v[i - n]) / (2. * dy)) -
			pow((u[i + 1] - u[i - 1]) / (2. * dx), 2) -
			2. * ((u[i + n] - u[i - n]) / (2. * dy) *
				(v[i + 1] - v[i - 1]) / (2. * dx)) -
			pow((v[i + n] - v[i - n]) / (2. * dy), 2));
	}
	//pressure_poisson
	for (int count = 0; count < nit; count++)
	{
		switch (position)
		{
		case 0: //inside
			p[i] = (((pn[i + 1] + pn[i - 1]) * pow(dy, 2) +
				(pn[i + n] + pn[i - n]) * pow(dx, 2)) /
				(2. * (pow(dx, 2) + pow(dy, 2))) -
				pow(dx, 2) * pow(dy, 2) / (2 * (pow(dx, 2) + pow(dy, 2))) *
				b[i]);
			break;
		case 1: //bottom
			p[i] = p[i + n];
			break;
		case 2: //top
			p[i] = 0.;
			break;
		case 3: //left
			p[i] = p[i + 1];
			break;
		case 4: //right
			p[i] = p[i - 1];
			break;
		default:
			break;
		}
	}
	//cavity_flow
	switch (position)
	{
	case 0: //inside
		u[i] = un[i] -
			un[i] * dt / dx *
			(un[i] - un[i - 1]) -
			vn[i] * dt / dy *
			(un[i] - un[i - n]) -
			dt / (2 * rho * dx) * (p[i + 1] - p[i - 1]) +
			nu * (dt / pow(dx, 2) *
				(un[i + 1] - 2 * un[i] + un[i - 1]) +
				dt / pow(dy, 2) *
				(un[i + n] - 2 * un[i] + un[i - n]));
		v[i] = vn[i] -
			un[i] * dt / dx *
			(vn[i] - vn[i - 1]) -
			vn[i] * dt / dy *
			(vn[i] - vn[i - n]) -
			dt / (2 * rho * dy) * (p[i + n] - p[i - n]) +
			nu * (dt / pow(dx, 2) *
				(vn[i + 1] - 2 * vn[i] + vn[i - 1]) +
				dt / pow(dy, 2) *
				(vn[i + n] - 2 * vn[i] + vn[i - n]));
		break;
	case 1: case 2: case 3: //bottom top left
		u[i] = 0.;
		v[i] = 0.;
		break;
	case 4: //right
		u[i] = 1.;
		v[i] = 0.;
		break;
	default:
		break;
	}
}

void matrix_sum(const double* m, int n)
{
	double sum = 0.;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			sum += m[i * n + j];
		}
	}
	std::cout << sum << std::endl;
}

int main()
{
	const int nit = 50;
	const int n = 40, N = n * n, M = 512;
	const int nt = 500;
	const double dx = 2. / (n - 1), dy = 2. / (n - 1);
	const double rho = 1.;
	const double nu = 0.02, dt = 0.01;
	//initial
	double* u, * v, * p, * b;
	double* un, * vn, * pn;
	cudaMallocManaged(&u, n * n * sizeof(double));
	cudaMallocManaged(&v, n * n * sizeof(double));
	cudaMallocManaged(&p, n * n * sizeof(double));
	cudaMallocManaged(&b, n * n * sizeof(double));
	cudaMallocManaged(&un, n * n * sizeof(double));
	cudaMallocManaged(&vn, n * n * sizeof(double));
	cudaMallocManaged(&pn, n * n * sizeof(double));
	for (int i = 0; i < n * n; i++)
	{
		u[i] = 0.;
		v[i] = 0.;
		p[i] = 0.;
		b[i] = 0.;
	}
	//cavity_flow
	for (int count = 0; count < nt; count++)
	{
		cudaMemcpy(un, u, n * n * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(vn, v, n * n * sizeof(double), cudaMemcpyHostToDevice);
		cavity_flow <<<(N + M - 1) / M, M >>> (u, v, p, b, un, vn, pn, dx, dy, dt, rho, nu, nit, n);
		cudaDeviceSynchronize();
		//printf("?\n");
	}
	matrix_sum(u, n);
	matrix_sum(v, n);
	cudaFree(u);
	cudaFree(v);
	cudaFree(p);
	cudaFree(b);
	cudaFree(un);
	cudaFree(vn);
	cudaFree(pn);
	return 0;
}
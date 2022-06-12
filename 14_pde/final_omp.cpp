#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

typedef std::vector<std::vector<double>> Matrix;

std::vector<double> line_space(int start, int end, int num);
void matrix_sum(const Matrix& m);

const int nit = 50;
const int n = 40;

int main()
{
    //initial
    int nt = 10;
    double dx = 2. / (n - 1), dy = 2. / (n - 1);
    std::vector<double> x = line_space(0, 2, n);
    std::vector<double> y = line_space(0, 2, n);
    //physical variables
    double rho = 1.;
    double nu = 0.1, dt = 0.01;
    Matrix u(n, std::vector<double>(n, 0.));
    Matrix v(n, std::vector<double>(n, 0.));
    Matrix p(n, std::vector<double>(n, 1.));
    Matrix b(n, std::vector<double>(n, 0.));
    auto tic = std::chrono::steady_clock::now();
#pragma omp parallel
    for (int count = 0; count < nt; count++)
    {
        Matrix un(u);
        Matrix vn(v);
        //build_up_b
#pragma omp for collapse(2)
        for(int i = 1; i < n-1; i++)
        {
            for (int j = 1; j < n-1; j++)
            {
                b[i][j] = rho * (1. / dt *
                                 ((u[i][j+1] - u[i][j-1]) /
                                  (2. * dx) + (v[i+1][j] - v[i-1][j]) / (2. * dy)) -
                                 pow((u[i][j+1] - u[i][j-1]) / (2. * dx), 2) -
                                 2. * ((u[i+1][j] - u[i-1][j]) / (2. * dy) *
                                       (v[i][j+1] - v[i][j-1]) / (2. * dx)) -
                                 pow((v[i+1][j] - v[i-1][j]) / (2. * dy), 2));
                //std::cout << b[i][j] << ' ';
            }
        }
        //pressure_poisson
        for (int count = 0; count < nit; count++)
        {
            Matrix pn(p);
#pragma omp for collapse(2)
            for (int i = 1; i < n-1; i++)
            {
                for (int j = 1; j < n-1; j++)
                {
                    p[i][j] = (((pn[i][j+1] + pn[i][j-1]) * pow(dy, 2) +
                                (pn[i+1][j] + pn[i-1][j]) * pow(dx, 2)) /
                               (2. * (pow(dx, 2) + pow(dy, 2))) -
                               pow(dx, 2) * pow(dy, 2) / (2 * (pow(dx, 2) + pow(dy, 2))) *
                               b[i][j]);
                    //std::cout << p[i][j] << ' ';
                }
            }
#pragma omp for
            for (int i = 0; i < n; i++)
            {
                p[i][n-1] = p[i][n-2]; // dp/dx = 0 at x = 2
                p[0][i] = p[1][i];   // dp/dy = 0 at y = 0
                p[i][0] = p[i][1];   // dp/dx = 0 at x = 0
                p[n-1][i] = 0;        // p = 0 at y = 2
            }
        }
        //cavity_flow
#pragma omp for collapse(2)
        for (int i = 1; i < n-1; i++)
        {
            for (int j = 1; j < n-1; j++)
            {
                u[i][j] = un[i][j] -
                          un[i][j] * dt / dx *
                          (un[i][j] - un[i][j-1]) -
                          vn[i][j] * dt / dy *
                          (un[i][j] - un[i-1][j]) -
                          dt / (2 * rho * dx) * (p[i][j+1] - p[i][j-1]) +
                          nu * (dt / pow(dx,2) *
                                (un[i][j+1] - 2 * un[i][j] + un[i][j-1]) +
                                dt / pow(dy, 2) *
                                (un[i+1][j] - 2 * un[i][j] + un[i-1][j]));
                v[i][j] = vn[i][j] -
                          un[i][j] * dt / dx *
                          (vn[i][j] - vn[i][j-1]) -
                          vn[i][j] * dt / dy *
                          (vn[i][j] - vn[i-1][j]) -
                          dt / (2 * rho * dy) * (p[i+1][j] - p[i-1][j]) +
                          nu * (dt / pow(dx,2) *
                                (vn[i][j+1] - 2 * vn[i][j] + vn[i][j-1]) +
                                dt / pow(dy, 2) *
                                (vn[i+1][j] - 2 * vn[i][j] + vn[i-1][j]));
                //std::cout << u[i][j] << ' ';
            }
        }
#pragma omp for
        for (int i = 0; i < n; i++)
        {
            u[0][i]  = 0;
            u[i][0]  = 0;
            u[i][n-1] = 0;
            u[n-1][i] = 1;   //set velocity on cavity lid equal to 1
            v[0][i]  = 0;
            v[n-1][i] = 0;
            v[i][0]  = 0;
            v[i][n-1] = 0;
        }
    }
    auto toc = std::chrono::steady_clock::now();
    double time = std::chrono::duration<double>(toc - tic).count();
    //matrix_sum(u);
    //matrix_sum(v);
    std::cout << "time:" << time << std::endl;
}

std::vector<double> line_space(int start, int end, int num)
{
    std::vector<double> res;
    if (num == 0)
    {
        return res;
    }
    if (num == 1)
    {
        res.push_back(start);
        return res;
    }
    double delta = (end - start) / (num - 1.);
    for (int i = 0; i < num-1; i++)
    {
        res.push_back(start + delta * i);
    }
    return res;
}

void matrix_sum(const Matrix& m)
{
    double sum = 0.;
    for (auto & i : m)
    {
        for (double j : i)
        {
            sum += j;
        }
    }
    std::cout << sum << std::endl;
}
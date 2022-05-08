#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

__global__ void bucket_in(int *key, int *bucket)
{
    int id = threadIdx.x;
    atomicAdd(&bucket[key[id]], 1);
}

__global__ void bucket_out(int *key, int *bucket)
{
    int id = threadIdx.x, start = 0;
    for (int k = 0; k < i; k++)
    {
        start += bucket[k];
    }
    for (; bucket[i]>0; bucket[i]--)
    {
        key[start++] = i;
    }
}

int main()
{
    int n = 50;
    int range = 5;
    int *key, *bucket;

    cudaMallocManaged(&key, n*sizeof(int));
    cudaMallocManaged(&bucket, range*sizeof(int));
    for (int i = 0; i < n; i++)
    {
        key[i] = rand() % range;
        std::cout << key[i] << ' ';
    }
    std::cout << std::endl;
    for (int i = 0; i < range; i++)
    {
        bucket[i] = 0;
    }
    bucket_in<<<1,n>>>(key, bucket);
    bucket_out<<<1,range>>>(key, bucket);
    cudaDeviceSynchronize();

    for (int i = 0; i < n; i++)
    {
        std::cout << key[i] << ' ';
    }
    cudaFree(key);
    cudaFree(bucket);
}
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m256 x_vec = _mm256_load_ps(x);
  __m256 y_vec = _mm256_load_ps(y);
  __m256 m_vec = _mm256_load_ps(m);
  __m256 zero = _mm256_set1_ps(0);
  __m256 fx_vec = _mm256_load_ps(fx);
  __m256 fy_vec = _mm256_load_ps(fy);

  for(int i=0; i<N; i++) {
      /*for(int j=0; j<N; j++) {
      if(i != j) {
          float rx = x[i] - x[j];
          float ry = y[i] - y[j];
          float r = std::sqrt(rx * rx + ry * ry);
          fx[i] -= rx * m[j] / (r * r * r);
          fy[i] -= ry * m[j] / (r * r * r);
        }

    }*/
    //rx
    __m256 xi = _mm256_set1_ps(x[i]);
    //ry
    __m256 yi = _mm256_set1_ps(y[i]);
    //(rx * rx + ry * ry)
    __m256 x_rest = _mm256_sub_ps(xi, x_vec);
    __m256 y_rest = _mm256_sub_ps(yi, y_vec);
    __m256 rx_2 = _mm256_mul_ps(x_rest, x_rest);
    __m256 ry_2 = _mm256_mul_ps(y_rest, y_rest);
    __m256 r1 = _mm256_add_ps(rx_2, ry_2);
    //condition
    __m256 mask = _mm256_cmp_ps(r1, zero, _CMP_GT_OQ);
    //(1/r)^3
    __m256 r2 = _mm256_rsqrt_ps(r1);
    r2 = _mm256_blendv_ps(zero, r2, mask);
    __m256 r3 = _mm256_mul_ps(r2, r2);
    r3 = _mm256_mul_ps(r3, r2);
    //fxi,fyi
    __m256 fxi = _mm256_mul_ps(x_rest, m_vec);
    fxi = _mm256_mul_ps(fxi, r3);
    __m256 fyi = _mm256_mul_ps(y_rest, m_vec);
    fyi = _mm256_mul_ps(fyi, r3);
    //result
    fx_vec = _mm256_sub_ps(fx_vec, fxi);
    fy_vec = _mm256_sub_ps(fy_vec, fyi);
    _mm256_store_ps(fx, fx_vec);
    _mm256_store_ps(fy, fy_vec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}

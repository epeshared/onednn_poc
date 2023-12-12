#include <stdlib.h>
#include <cstring>
#include <immintrin.h>
#include <random>
#include "oneapi/dnnl/dnnl.hpp"
#include <sys/time.h>

static bool is_avx512_enabled() {
  char* env = getenv("AVX512_ENABLE");
  bool ret;
  (env != NULL && std::strcmp(env, "1") == 0) ? ret = true: ret = false;
  return ret;
}

float fvec_inner_product_avx512(const float* x, const float* y, size_t d) {    
    int mod_16_num = d / 16;    
    float sum = 0.0F; 
    size_t i = 0;
    if (mod_16_num > 0) {
      __m512 res = _mm512_setzero_ps();      
      for (; i < mod_16_num*16; i += 16) {
          __m512 vx = _mm512_loadu_ps(x + i);
          __m512 vy = _mm512_loadu_ps(y + i);
          res = _mm512_fmadd_ps(vx, vy, res);
      }
      sum = _mm512_reduce_add_ps(res);
    }

    for (; i < d; ++i) {
      sum += x[i] * y[i];
    }

    return sum;
}

float fvec_inner_product(float *x, float *y, size_t d) {
  float res = 0.0F;
  for (size_t i = 0; i < d; ++i) {
    res += x[i] * y[i];
    // printf("%d ", res);
  }
  return res;
}

void print_f32(float *X_f32, size_t row, size_t col)
{
  for (size_t i = 0; i < row * col; i++) {
    if (i % col == 0 && i != 0) {
      printf("\n");
    }
    printf("%0.2f ", X_f32[i]);
  }
  printf("\n");
}

#define VERIFY 1
#define SCALAR_TEST 1
#define AVX512_TEST 1
#define PERFORMANCE_TEST 1

int main(int argc, char **argv){
  uint32_t DIM = atoi(argv[1]);
 
  uint32_t xrow = 5;
  uint32_t xcol = DIM;
 
  uint32_t yrow = 10;
  uint32_t ycol = DIM;
 
  uint32_t loop = 100000;
  float *x = (float *)malloc(xrow * xcol * sizeof(float));
  float *y = (float *)malloc(yrow * ycol * sizeof(float));
  float *ret2 = (float *)malloc(xrow * yrow * sizeof(float));
  float *ret1 = (float *)malloc(xrow * yrow * sizeof(float));
 
  srand(time(NULL));
  uint32_t i, j;
  float *x_i;
  float *y_j;

    // 创建一个随机数生成器
    std::default_random_engine generator;
    // 创建一个均匀分布的浮点数分布，范围从0.0到1.0
    std::uniform_real_distribution<float> distribution(0.0, 1.0);  
 
  for (i = 0; i < xrow * xcol; i++)
    x[i] = distribution(generator);
  for (i = 0; i < yrow * ycol; i++)
    y[i] = distribution(generator);
  for (i = 0; i < xrow * yrow; i++)
    ret2[i] = 0;
  for (i = 0; i < xrow * yrow; i++)
    ret1[i] = 0;
 
  // printf("----------- matrix x -----------------------\n");
  // print_s8(x, xrow, xcol);
 
  // printf("----------- matrix y -----------------------\n");
  // print_s8(y, yrow, ycol);
 
  struct timeval start, end;
  long seconds, microseconds;
  double elapsed;
 
#ifdef SCALAR_TEST
  gettimeofday(&start, NULL);
#pragma omp parallel num_threads(144)
  for (uint32_t k = 0; k < loop; k++) {
#pragma omp for    
    for (i = 0; i < xrow; i++) {
      x_i = x + i * DIM;
      for (j = 0; j < yrow; j++) {
        y_j = y + j * DIM;
        ret1[i * yrow + j] = fvec_inner_product(x_i, y_j, DIM);
      }
    }
  }
 
  gettimeofday(&end, NULL);
  printf("----------- scalar result -----------------------\n");
  print_f32(ret1, xrow, yrow);
  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  elapsed = seconds + microseconds * 1e-6;
  printf("Function Scalar took %f seconds to execute.\n", elapsed);
  printf("\n");
#endif
 
  // for (uint32_t i = 0; i < xrow; i++)
  // {
  //   for (uint32_t j = 0; j < yrow; j++)
  //   {
  //     ret1[i * yrow + j] = 0;
  //   }
  // }
 
  // for (uint32_t i = 0; i < xrow; i++)
  // {
  //   for (uint32_t j = 0; j < yrow; j++)
  //   {
  //     for (uint32_t k = 0; k < xcol; k++)
  //     {
  //       ret1[i * yrow + j] += x[i * xcol + k] * y[j * xcol + k];
  //     }
  //   }
  // }
  // printf("----------- default scalar result -----------------------\n");
  // print_s32(ret1, xrow, yrow);
 
#ifdef AVX512_TEST
  gettimeofday(&start, NULL);
// #pragma omp parallel num_threads(144)
  for (uint32_t k = 0; k < loop; k++) {
// #pragma omp for   
    for (i = 0; i < xrow; i++) {
      x_i = x + i * DIM;
      for (j = 0; j < yrow; j++) {
        y_j = y + j * DIM;
        ret2[i * yrow + j] = fvec_inner_product_avx512(x_i, y_j, DIM);
      }
    }
  }
  gettimeofday(&end, NULL);
  printf("----------- avx512 result -----------------------\n");
  print_f32(ret2, xrow, yrow);
  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  elapsed = seconds + microseconds * 1e-6;
  printf("Function DNN took %f seconds to execute.\n", elapsed);
  printf("\n");
#endif
 
#ifdef VERIFY
  for (i = 0; i < xrow * ycol; i++)
  {
    if (ret2[i] != ret1[i])
    {
      printf("ERROR!the %u element is not equal, %0.6f:%0.6f\n", i, ret2[i], ret1[i]);
    }
    break;
  }
#endif  
}



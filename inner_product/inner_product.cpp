#include <immintrin.h>
#include "oneapi/dnnl/dnnl.hpp"
#include <sys/time.h>
 
using namespace dnnl;
// using tag = memory::format_tag;
// using dt = memory::data_type;
 
memory src_mem;
memory dst_mem;
memory weights_mem;
 
// inner_product_forward inner_product_prim;
// std::unordered_map<int, memory> inner_product_args;

void init_onednn_inner_product(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol, engine &cpu_engine,
                               inner_product_forward &inner_product_prim, inner_product_forward::primitive_desc &inner_product_pd)
{
  memory::desc src_md({xrow, xcol}, memory::data_type::s8, memory::format_tag::any);
  memory::desc inner_product_weights_md({yrow, ycol}, memory::data_type::s8, memory::format_tag::any);
  memory::desc dst_md({xrow, yrow}, memory::data_type::s32, memory::format_tag::ab);
 
  // Create inner product primitive descriptor.
  inner_product_pd = inner_product_forward::primitive_desc(
      cpu_engine,
      prop_kind::forward_training,
      src_md, inner_product_weights_md, dst_md);
 
  inner_product_prim = inner_product_forward(inner_product_pd);
}
 
// void init_onednn_inner_product(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol, engine &cpu_engine,
//                                inner_product_forward &inner_product_prim, inner_product_forward::primitive_desc &inner_product_pd)
// {
//   memory::desc src_md({xrow, xcol}, memory::data_type::s8, memory::format_tag::ab);
//   memory::desc inner_product_weights_md({yrow, ycol}, memory::data_type::s8, memory::format_tag::AB16b32a4b);
//   memory::desc dst_md({xrow, yrow}, memory::data_type::s32, memory::format_tag::ab);
 
//   // Create inner product primitive descriptor.
//   inner_product_pd = inner_product_forward::primitive_desc(
//       cpu_engine,
//       prop_kind::forward_training,
//       src_md, inner_product_weights_md, dst_md);
 
//   inner_product_prim = inner_product_forward(inner_product_pd);
// }
 
int32_t inner_product_dnn(int8_t *x, int8_t *y, int32_t *z,
                          engine &cpu_engine,
                          stream &engine_stream,
                          inner_product_forward &inner_product_prim,
                          inner_product_forward::primitive_desc &inner_product_pd)
{
 
  // int32_t ret = 0;
  memory src_mem = memory(inner_product_pd.src_desc(), cpu_engine, x);
  memory dst_mem = memory(inner_product_pd.dst_desc(), cpu_engine, z);
  memory user_wei_mem = memory({inner_product_pd.weights_desc().get_dims(),
                                memory::data_type::s8, memory::format_tag::ab},
                               cpu_engine, y);
 
  auto weights_mem = user_wei_mem;
 
  if (inner_product_pd.weights_desc() != user_wei_mem.get_desc())
  {
    weights_mem = memory(inner_product_pd.weights_desc(), cpu_engine);
    reorder(user_wei_mem, weights_mem)
        .execute(engine_stream, user_wei_mem,
                 weights_mem);
  }
  inner_product_prim.execute(engine_stream, {{DNNL_ARG_SRC, src_mem},
                                             {DNNL_ARG_WEIGHTS, weights_mem},
                                             {DNNL_ARG_DST, dst_mem}});
 
  // Wait for the computation to finalize.
  engine_stream.wait();
 
  return 0;
}
 
int32_t fvec_inner_product(int8_t *x, int8_t *y, size_t d)
{
  int32_t res = 0;
  for (size_t i = 0; i < d; ++i)
  {
    res += x[i] * y[i];
    // printf("%d ", res);
  }
  return res;
}
 
#define VERIFY 1
#define OMP_TEST 1
#define ONEDNN_TEST 1
#define PERFORMANCE_TEST 1
 
void print_s32(int32_t *X_f32, size_t row, size_t col)
{
  for (size_t i = 0; i < row * col; i++)
  {
    if (i % col == 0 && i != 0)
    {
      printf("\n");
    }
    printf("%d ", X_f32[i]);
  }
  printf("\n");
}
 
void print_s8(int8_t *X_s8, size_t row, size_t col)
{
  for (size_t i = 0; i < row * col; i++)
  {
    if (i % col == 0 && i != 0)
    {
      printf("\n");
    }
    printf("%d ", X_s8[i]);
  }
  printf("\n");
}
 
int main(int argc, char **argv)
{
  uint32_t DIM = atoi(argv[1]);
 
  uint32_t xrow = 1;
  uint32_t xcol = DIM;
 
  uint32_t yrow = 1000000;
  uint32_t ycol = DIM;
 
  uint32_t loop = 100;
  int8_t *x = (int8_t *)malloc(xrow * xcol * sizeof(int8_t));
  int8_t *y = (int8_t *)malloc(yrow * ycol * sizeof(int8_t));
  int32_t *ret2 = (int32_t *)malloc(xrow * yrow * sizeof(int32_t));
  int32_t *ret1 = (int32_t *)malloc(xrow * yrow * sizeof(int32_t));
 
  srand(time(NULL));
  uint32_t i, j;
  int8_t *x_i;
  int8_t *y_j;
 
  for (i = 0; i < xrow * xcol; i++)
    x[i] = i+1;//(int8_t)rand() % 256 - 128;
  for (i = 0; i < yrow * ycol; i++)
    y[i] = i + 1; //(int8_t)rand() % 256 - 128;
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
 
#ifdef OMP_TEST
  gettimeofday(&start, NULL);
#pragma omp parallel num_threads(144)
#pragma omp for
  for (uint32_t k = 0; k < loop; k++)
  {
    for (i = 0; i < xrow; i++)
    {
      x_i = x + i * DIM;
      for (j = 0; j < yrow; j++)
      {
        y_j = y + j * DIM;
        ret1[i * DIM + j] = fvec_inner_product(x_i, y_j, DIM);
      }
    }
  }
 
  gettimeofday(&end, NULL);
  // printf("----------- scalar result -----------------------\n");
  // print_s32(ret1, xrow, yrow);
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
 
#ifdef ONEDNN_TEST
  dnnl::engine cpu_engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(cpu_engine);
  inner_product_forward inner_product_prim;
  inner_product_forward::primitive_desc inner_product_pd;
  init_onednn_inner_product(xrow, xcol, yrow, ycol, cpu_engine, inner_product_prim, inner_product_pd);
  gettimeofday(&start, NULL);
  for (uint32_t k = 0; k < loop; k++)
  {
    inner_product_dnn(x, y, ret2, cpu_engine, engine_stream,
                      inner_product_prim, inner_product_pd);
    // creat_onednn_inner_product(xrow, xcol, yrow, ycol, x, y, ret2, engine_stream, inner_product_prim, inner_product_pd);
  }
  gettimeofday(&end, NULL);
  // printf("----------- dnnl result -----------------------\n");
  // print_s32(ret2, xrow, yrow);
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
      printf("ERROR!the %u element is not equal\n", i);
    }
    break;
  }
#endif
}
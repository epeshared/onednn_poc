#include <immintrin.h>
#include <random>
#include "oneapi/dnnl/dnnl.hpp"
#include <sys/time.h>
 
// using tag = memory::format_tag;
// using dt = memory::data_type;
 
dnnl::memory f32_mem1;
dnnl::memory f32_mem2;
dnnl::memory bf16_mem1;
dnnl::memory bf16_mem2;
dnnl::memory f32_dst_mem;
 
// inner_product_forward inner_product_prim;
// std::unordered_map<int, memory> inner_product_args;
 
void print_f32(float *X_f32, size_t row, size_t col)
{
  for (size_t i = 0; i < row * col; i++)
  {
    if (i % col == 0 && i != 0)
    {
      printf("\n");
    }
    printf("%f ", X_f32[i]);
  }
  printf("\n");
}
 
void comput_f32bf16f32(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol,
                       float *in_f32_1, float *in_f32_2, float *out_f32,
                       dnnl::engine &cpu_engine,
                       dnnl::stream &engine_stream)
{
 
  dnnl::memory::desc f32_md1({xrow, xcol}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  dnnl::memory::desc f32_md2({yrow, ycol}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  dnnl::memory::desc f32_dst_md2({xrow, yrow}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
 
  // 创建内存
  f32_mem1 = dnnl::memory(f32_md1, cpu_engine, in_f32_1);
  f32_mem2 = dnnl::memory(f32_md2, cpu_engine, in_f32_2);
  f32_dst_mem = dnnl::memory(f32_dst_md2, cpu_engine, out_f32);
 
 
  // inner memory bf16
  dnnl::memory::desc bf16_md1({xrow, xcol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any);
  dnnl::memory::desc bf16_md2({yrow, ycol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any);
 
  dnnl::inner_product_forward::primitive_desc inner_product_pd;
  inner_product_pd = dnnl::inner_product_forward::primitive_desc(
      cpu_engine, dnnl::prop_kind::forward_training,
      bf16_md1, bf16_md2, f32_dst_md2);
 
  auto bf16_mem1 = dnnl::memory(inner_product_pd.src_desc(), cpu_engine);
  dnnl::reorder(f32_mem1, bf16_mem1).execute(engine_stream, f32_mem1, bf16_mem1);
 
  auto bf16_mem2 = dnnl::memory(inner_product_pd.weights_desc(), cpu_engine);
  dnnl::reorder(f32_mem2, bf16_mem2).execute(engine_stream, f32_mem2, bf16_mem2);
 
  auto inner_product_prim = dnnl::inner_product_forward(inner_product_pd);
  inner_product_prim.execute(engine_stream, {{DNNL_ARG_SRC, bf16_mem1},
                                             {DNNL_ARG_WEIGHTS, bf16_mem2},
                                             {DNNL_ARG_DST, f32_dst_mem}});
 
  // Wait for the computation to finalize.
  engine_stream.wait();
 
  printf("comput_f32bf16f32 finished\n");
}
 
int main(int argc, char **argv)
{
  uint32_t DIM = 1024;
 
  uint32_t xrow = 64;
  uint32_t xcol = DIM;
 
  uint32_t yrow = 100;
  uint32_t ycol = DIM;
 
  uint32_t loop = 1;
  float *x = (float *)malloc(xrow * xcol * sizeof(float));
  float *y = (float *)malloc(yrow * ycol * sizeof(float));
  float *ret = (float *)malloc(xrow * yrow * sizeof(float));
 
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
 
  int i;
  for (i = 0; i < xrow * xcol; i++)
    x[i] = dis(gen);
  // printf("-----------------x--------------\n");
  // print_f32(x, xrow , xcol);
 
  for (i = 0; i < yrow * ycol; i++)
    y[i] = dis(gen);
  // printf("-----------------y--------------\n");
  // print_f32(y, yrow , ycol);
 
  for (i = 0; i < xrow * yrow; i++)
    ret[i] = 0.0f;
 
  printf("init data finished\n");
 
  dnnl::engine cpu_engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream engine_stream(cpu_engine);
  comput_f32bf16f32(xrow, xcol, yrow, ycol, x, y, ret, cpu_engine, engine_stream);
 
  // printf("-----------------ret--------------\n");
  // print_f32(ret, xrow , yrow);
}
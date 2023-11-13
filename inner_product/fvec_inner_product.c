#include <immintrin.h>
#include "oneapi/dnnl/dnnl.hpp"
#include <sys/time.h>

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

memory src_mem;
memory dst_mem;
memory weights_mem;
memory ret_mem;

inner_product_forward inner_product_prim;
std::unordered_map<int, memory> inner_product_args;

dnnl::engine cpu_engine(dnnl::engine::kind::cpu, 0);
dnnl::stream engine_stream(cpu_engine);

void init_onednn(uint32_t DIM) {
  const memory::dim N = 1, // batch size
          IC = 1, // input channels
          IH = DIM, // tensor height
          IW = DIM, // tensor width
          OC = 1; // output channels

  memory::desc src_md({DIM, DIM}, memory::data_type::s8, memory::format_tag::ab);
  memory::desc inner_product_weights_md({DIM, DIM}, memory::data_type::s8, memory::format_tag::ab);
  memory::desc dst_md({DIM, DIM}, memory::data_type::s32, memory::format_tag::ab);
  memory::desc ret_md({DIM, 1}, memory::data_type::s32, memory::format_tag::ab);

  // Create inner product primitive descriptor.
  inner_product_forward::primitive_desc inner_product_pd = inner_product_forward::primitive_desc(
        cpu_engine,
        prop_kind::forward_training, 
        src_md, inner_product_weights_md,dst_md);
  
  src_mem = memory(src_md, cpu_engine, NULL);
  dst_mem = memory(dst_md, cpu_engine, NULL);
  weights_mem = memory(inner_product_weights_md, cpu_engine, NULL);
  ret_mem = memory(ret_md, cpu_engine, NULL);
 
  inner_product_prim = inner_product_forward(inner_product_pd);

  // Primitive arguments.
  
  inner_product_args.insert({DNNL_ARG_SRC, src_mem});
  inner_product_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
  inner_product_args.insert({DNNL_ARG_DST, dst_mem});          
}

int32_t inner_product_dnn(int8_t* x, int8_t* y, int32_t* z,
        memory& x_mem, memory& y_mem, memory& dst_mem, dnnl::stream engine_stream, 
        std::unordered_map<int, memory>& inner_product_args,
        inner_product_forward& inner_product_prim) {

    // int32_t ret = 0;
    x_mem.set_data_handle(x);
    y_mem.set_data_handle(y);
    dst_mem.set_data_handle(z);

    inner_product_prim.execute(engine_stream, inner_product_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    return 0;
}

int32_t fvec_inner_product(int8_t* x, int8_t* y, size_t d) {
    int32_t res = 0;
    for (size_t i = 0; i < d; ++i) {
        res += x[i] * y[i];
        // printf("%d ", res);
    }
    return res;
}

int main(int argc, char **argv){
  uint32_t DIM = 1024;
  uint32_t loop = 10;
  int8_t* x = (int8_t*)malloc(DIM * DIM * sizeof(int8_t));
  int8_t* y = (int8_t*)malloc(DIM * DIM * sizeof(int8_t));
  int32_t* ret2 = (int32_t*)malloc(DIM * DIM * sizeof(int32_t));
  int32_t* ret1 = (int32_t*)malloc(DIM * DIM * sizeof(int32_t));

  srand(time(NULL));
  uint32_t i,j;
  int8_t* x_i;
  int8_t* y_j;   

  for (i = 0; i < DIM*DIM; i++) x[i] = (int8_t)rand() % 256 - 128;
  for (i = 0; i < DIM*DIM; i++) y[i] = (int8_t)rand() % 256 - 128;
  for (i = 0; i < DIM*DIM; i++) ret2[i] = 0;
  for (i = 0; i < DIM*DIM; i++) ret1[i] = 0;


  struct timeval start, end;
  // int32_t ret1[DIM][DIM];
  // int32_t ret2[DIM][DIM];

  gettimeofday(&start, NULL);
#pragma omp parallel num_threads(144)
#pragma omp for
  for (uint32_t k = 0 ; k < loop;k++) { 
    for (i = 0; i < DIM; i++) {
      x_i = x + i*DIM;
      for (j = 0; j < DIM; j++) {
        y_j = y + j*DIM;
        ret1[i*DIM + j] = fvec_inner_product(x_i,y_j, DIM);
        // printf("i:%d %d ",i,ret[i]);
      }
    }
  }
  gettimeofday(&end, NULL);
  printf("\n");
  long seconds = end.tv_sec - start.tv_sec;
  long microseconds = end.tv_usec - start.tv_usec;
  double elapsed = seconds + microseconds*1e-6;
  printf("Function Scalar took %f seconds to execute.\n", elapsed);
  
  
  init_onednn(DIM);
  gettimeofday(&start, NULL);
  for (uint32_t k = 0 ; k < loop;k++) {
    inner_product_dnn(x, y, ret2, src_mem, weights_mem, dst_mem, engine_stream, 
        inner_product_args, inner_product_prim);         
  }   
  gettimeofday(&end, NULL);
  printf("\n");
  for (i = 0; i < DIM*DIM; i++) {
    if (ret2[i] != ret1[i]) {
      printf("the %d element is not equal\n");
    }
    break;  
  }    

  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  elapsed = seconds + microseconds*1e-6;
  printf("Function DNN took %f seconds to execute.\n", elapsed);
}
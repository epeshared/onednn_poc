#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <stdlib.h>
#include <initializer_list>

// #include "dnn_util.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <sys/time.h>

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;


int32_t inner_product_dnn(int8_t* x, int8_t* y,
        memory& x_mem, memory& y_mem, memory& dst_mem, dnnl::stream engine_stream, 
        std::unordered_map<int, memory>& inner_product_args,
        inner_product_forward& inner_product_prim) {
    // Tensor dimensions.

    int32_t ret = 0;
    x_mem.set_data_handle(x);
    y_mem.set_data_handle(y);
    dst_mem.set_data_handle(&ret);

    inner_product_prim.execute(engine_stream, inner_product_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    return ret;
}


int32_t fvec_inner_product(int8_t* x, int8_t* y, size_t d) {
    int32_t res = 0;
    for (size_t i = 0; i < d; ++i) {
        res += x[i] * y[i];
        // printf("%d ", res);
    }
    return res;
} 

#define MAX 1024
#define MAX_ROWS 16
#define MAX_COLS 64

int main(int argc, char **argv){
  int DIM = 1024;
  int8_t x[DIM][DIM];
  int8_t y[DIM][DIM];
  // int8_t z[DIM];

  // int rows = DIM / MAX_COLS;
  // int left = DIM % MAX_COLS;

  // int8_t random_nums[DIM];

  // 初始化随机数生成器
  srand(time(NULL));
  int i,j;

  for (i = 0; i < DIM; i++) {
    for (j = 0; j < DIM; j++) {
        x[i][j] = (int8_t)rand() % 256 - 128; // 生成-128到127的随机数
    }
    
  }

  for (i = 0; i < DIM; i++) {
    for (j = 0; j <DIM; j++) {
        y[i][j] = (int8_t)rand() % 256 - 128; // 生成-128到127的随机数
    }    
  }    

  struct timeval start, end;
  int32_t ret;

  gettimeofday(&start, NULL); 
  for (j = 0 ; j < 1024;j++) {
    for (i = 0;i<DIM;i++) {
      ret = fvec_inner_product(x[i], y[i], DIM);
      // printf("%d ",ret);
    }    
  }
  gettimeofday(&end, NULL);
  // printf("\n");
  long seconds = end.tv_sec - start.tv_sec;
  long microseconds = end.tv_usec - start.tv_usec;
  double elapsed = seconds + microseconds*1e-6;
  printf("Function Scalar took %f seconds to execute.\n", elapsed);
  
  dnnl::engine engine(dnnl::engine::kind::cpu, 0);

  // Create dnnl::stream.
  dnnl::stream engine_stream(engine);
  const memory::dim N = 1, // batch size
          IC = DIM, // input channels
          IH = 1, // tensor height
          IW = 1, // tensor width
          OC = 1; // output channels

  // Source (src), weights, bias, and destination (dst) tensors
  // dimensions.
  memory::dims src_dims = {N, IC}; //, IH, IW};
  memory::dims weights_dims = {OC, IC}; //, IH, IW};
  memory::dims dst_dims = {N, OC};

  memory::desc src_md = memory::desc(src_dims, dt::s8, tag::nc);
  memory::desc dst_md = memory::desc(dst_dims, dt::s32, tag::nc);
  //memory::desc inner_product_weights_md = memory::desc(weights_dims, dt::s8, tag::any);
  memory::desc inner_product_weights_md = memory::desc(weights_dims, dt::s8, tag::oi);


//   const float alpha = 0;
//   const float beta = 0;
//   post_ops inner_product_ops;
  // inner_product_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
  //primitive_attr inner_product_attr;
  //inner_product_attr.set_post_ops(inner_product_ops);

  // Create inner product primitive descriptor.
  inner_product_forward::primitive_desc inner_product_pd = inner_product_forward::primitive_desc(engine,
          prop_kind::forward_training, src_md, inner_product_weights_md,
          /*bias_md,*/ dst_md); //, inner_product_attr);  
  
  memory src_mem = memory(src_md, engine, NULL);
  memory dst_mem = memory(dst_md, engine, NULL);
  //memory weights_mem = memory(inner_product_pd.weights_desc()/*inner_product_weights_md*/, engine, NULL);
  memory weights_mem = memory(inner_product_weights_md, engine, NULL);

  inner_product_forward inner_product_prim = inner_product_forward(inner_product_pd);

  // Primitive arguments.
  std::unordered_map<int, memory> inner_product_args;
  inner_product_args.insert({DNNL_ARG_SRC, src_mem});
  inner_product_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
  inner_product_args.insert({DNNL_ARG_DST, dst_mem});          
  
  for (i = 0; i< 10; i++) {
    inner_product_dnn(x[i], y[i], src_mem, weights_mem, dst_mem, engine_stream, inner_product_args, inner_product_prim);
  }

  gettimeofday(&start, NULL);
  for (j = 0 ; j < 1024;j++) {
    for (i = 0;i<DIM;i++) {
      ret = inner_product_dnn(x[i], y[i], src_mem, weights_mem, dst_mem, engine_stream, inner_product_args, inner_product_prim);
      // printf("%d ",ret);
    }
  }
  gettimeofday(&end, NULL);
  // printf("\n");
  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  elapsed = seconds + microseconds*1e-6;
  printf("Function DNN took %f seconds to execute.\n", elapsed);
}
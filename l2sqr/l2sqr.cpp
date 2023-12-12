#include <immintrin.h>
#include "oneapi/dnnl/dnnl.hpp"
#include <sys/time.h>
#include <iostream>

static dnnl::engine cpu_engine;
static dnnl::stream engine_stream;

static void comput_f32bf16f32_l2sqr(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol,
    float* in_f32_1, float* in_f32_2, float* out_f32) {

  dnnl::memory::desc f32_md1 = dnnl::memory::desc({xrow, xcol}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  dnnl::memory::desc f32_md2 = dnnl::memory::desc({yrow, ycol}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
  dnnl::memory::desc f32_dst_md2 = dnnl::memory::desc({xrow, yrow}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

  dnnl::memory f32_mem1 = dnnl::memory(f32_md1, cpu_engine, in_f32_1);
  dnnl::memory f32_mem2 = dnnl::memory(f32_md2, cpu_engine, in_f32_2);
  dnnl::memory f32_dst_mem = dnnl::memory(f32_dst_md2, cpu_engine, out_f32); 

  // inner memory bf16
  dnnl::memory::desc bf16_md1 = dnnl::memory::desc({xrow, xcol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any);
  dnnl::memory::desc bf16_md2 = dnnl::memory::desc({yrow, ycol}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::any); 

  dnnl::binary::primitive_desc binary_pd = 
          dnnl::binary::primitive_desc(cpu_engine, dnnl::algorithm::binary_mul, bf16_md1, bf16_md2, md_c); 
  
  
  dnnl::eltwise_forward::primitive_desc l2sqr_pd = dnnl::eltwise_forward::primitive_desc(
      cpu_engine, dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_square,
      bf16_md1, bf16_md2, f32_dst_md2);

  dnnl::eltwise_forward l2sqr_prim = dnnl::eltwise_forward(l2sqr_pd);

  dnnl::memory bf16_mem1 = dnnl::memory(inner_product_pd.src_desc(), cpu_engine);
  dnnl::reorder(f32_mem1, bf16_mem1).execute(engine_stream, f32_mem1, bf16_mem1);
 
  dnnl::memory bf16_mem2 = dnnl::memory(inner_product_pd.weights_desc(), cpu_engine);
  dnnl::reorder(f32_mem2, bf16_mem2).execute(engine_stream, f32_mem2, bf16_mem2);

  inner_product_prim.execute(engine_stream, {{DNNL_ARG_SRC, bf16_mem1},
                                             {DNNL_ARG_WEIGHTS, bf16_mem2},
                                             {DNNL_ARG_DST, f32_dst_mem}});
 
  // Wait for the computation to finalize.
  engine_stream.wait();
 
  // printf("comput_f32bf16f32_inner_product finished#######>\n");
}

int main() {
    // Example usage:
    size_t m = 3; // Number of rows in X
    size_t n = 4; // Number of rows in Y
    size_t d = 2; // Number of columns in X and Y

    std::vector<float> X = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<float> Y = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0};
    std::vector<float> result(m * n);

    dnnl::engine cpu_engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream(cpu_engine);

    compute_l2sqr_matrix(X.data(), Y.data(), m, n, d, result);

    // Print the result matrix
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << result[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

#include <immintrin.h>
#include <random>
#include <sys/time.h>
#include "oneapi/dnnl/dnnl.hpp"

static dnnl::memory int8_mem;
static dnnl::memory float_mem;

static dnnl::engine cpu_engine(dnnl::engine::kind::cpu, 0);
static dnnl::stream engine_stream(cpu_engine);

static dnnl::memory _scale_m;
static dnnl::memory zp_m;
static dnnl::reorder::primitive_desc q10n_pd;

void init_data(float *v, float min_value, float max_value, size_t size, float init_val = 0.0f) {
    if (init_val == 0.0f) {
        for (size_t i = 0; i < size; i++)
            v[i] = init_val;
    } else {
        std::mt19937 gen;
        std::uniform_real_distribution<float> u(min_value, max_value);        
        for (size_t i = 0; i < size; i++)
            v[i] = u(gen);
    }

}

static void init_onednn_q10n(int64_t row, int64_t col, float scale, int32_t zp) {
    dnnl::memory::desc float_md({row, col}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::ab);
    float_mem = dnnl::memory(float_md, cpu_engine, NULL); 

    dnnl::memory::desc int8_md({row, col}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::ab);
    int8_mem = dnnl::memory(int8_md, cpu_engine, NULL);

    dnnl::primitive_attr q10n_attr;
    q10n_attr.set_scales_mask(DNNL_ARG_DST, /* mask */ 0);
    q10n_attr.set_zero_points_mask(DNNL_ARG_DST, /* mask */ 0);

    dnnl::reorder::primitive_desc q10n_pd(cpu_engine, float_md, cpu_engine, int8_md, q10n_attr);
    _scale_m = dnnl::memory({{1}, dnnl::memory::data_type::f32, {1}}, cpu_engine, &scale);
    zp_m = dnnl::memory({{1}, dnnl::memory::data_type::s32, {1}}, cpu_engine, &zp);

}

void quantinzation(int8_t* _int_data, float* _float_data, dnnl::primitive_attr q10n_attr) {
    
    int8_mem.set_data_handle(_int_data);
    float_mem.set_data_handle(_float_data);

    dnnl::reorder(q10n_pd).execute(engine_stream,
            {{DNNL_ARG_SRC, float_mem}, {DNNL_ARG_DST, int8_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, _scale_m},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, zp_m}});

    engine_stream.wait();
}

int main() {
    const int64_t row_a = 10, col_a = 20;
    const int64_t row_b = col_a, col_b = 30;
    const int64_t row_c = row_a, col_c = col_b;

    float *A_f32, *B_f32, *C_f32;  

    // Data distribution for matrices A and B
    const float param_A_min_val = -2.f;
    const float param_A_max_val = 1.4f;

    const float param_B_min_val = -1.f;
    const float param_B_max_val = -param_B_min_val; // B is centered around 0      

    A_f32 = (float*)malloc(row_a * col_a * sizeof(float));
    B_f32 = (float*)malloc(row_b * col_b * sizeof(float)); 
    C_f32 = (float*)malloc(row_c * col_c * sizeof(float));  

    init_data(A_f32, param_A_min_val, param_A_max_val, row_a*col_a);  
    init_data(B_f32, param_B_min_val, param_B_max_val, row_b*col_b);  
}
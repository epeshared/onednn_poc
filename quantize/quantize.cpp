#include <immintrin.h>
#include <random>
#include <sys/time.h>
#include <cassert>
#include "oneapi/dnnl/dnnl.hpp"

dnnl::engine eng(dnnl::engine::kind::cpu, 0);

void init_vector(float* v, float min_value, float max_value, size_t size) {
    std::mt19937 gen;
    std::uniform_real_distribution<float> u(min_value, max_value);

    for (int i = 0; i < size; i++)
        v[i] = u(gen);
}

// void f32_matmul_compute(int64_t M, int64_t N, int64_t K,
//         const std::vector<float> &A_f32, const std::vector<float> &B_f32,
//         std::vector<float> &C_f32) {
//     // Initialize memory descriptors that describes matrices in Row-Major format
//     dnnl::memory::desc a_md({M, K}, dnnl::memory::data_type::f32, {K, 1});

//     // Wrap raw pointers into oneDNN memory objects
//     dnnl::memory A_f32_m(a_md, eng, (void *)A_f32.data());

//     // Create a MatMul primitive
//     dnnl::matmul::primitive_desc matmul_pd(eng, a_md, b_md, c_md);
//     dnnl::matmul matmul_p(matmul_pd);

//     dnnl::stream s(eng);
//     matmul_p.execute(s,
//             {{DNNL_ARG_SRC, A_f32_m}, {DNNL_ARG_WEIGHTS, B_f32_m},
//                     {DNNL_ARG_DST, C_f32_m}});
//     s.wait();
// }

template <typename T>
void find_min_max(T* v,size_t size, float &min_value, float &max_value) {
    min_value = max_value = v[0];
    for (int i = 0;i< size; i++) {        
        min_value = std::min<float>(min_value, v[i]);
        max_value = std::max<float>(max_value, v[i]);
    }
}

template <typename T>
void compute_q10n_params(const char *message, float* v, size_t size,
        float &scale, int32_t &zp) {
    // Find property of T integer type
    // Simple trick to improve accuracy: shrink the range a little bit
    float max_int = (float)std::numeric_limits<T>::max() - 1;
    float min_int = (float)std::numeric_limits<T>::lowest() + 1;

#ifndef OMIT_WORKAROUND_FOR_SKX
    // Read more in CPU / Section 1 here:
    // https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html
    if (std::is_same<T, uint8_t>::value) max_int /= 2;
#endif

    // Find min and max value in array
    float min_val = v[0], max_val = v[0];
    find_min_max(v, size, min_val, max_val);

    // Compute appropriate scale
    scale = (max_val - min_val) / (max_int - min_int);

    // Compute appropriate offset
    if (std::is_same<T, int8_t>::value)
        zp = 0;
    else
        zp = (int32_t)(max_int - max_val / scale);
    printf("\tComputing q10n params for %s\n"
           "\t\tData type: %s\n"
           "\t\tScale:%.3g (inverse scale:%.3g)\n"
           "\t\tZero point:%d\n\n",
            message, std::is_same<T, int8_t>::value ? "int8_t" : "uint8_t",
            scale, 1 / scale, zp);
}

void quantize(float* X_f32, float scale_X, int32_t zp_X,
        dnnl::memory &X_int_m) {
    using dt = dnnl::memory::data_type;

    dnnl::stream s(eng);

    dnnl::memory::desc x_int_md = X_int_m.get_desc();
    const auto &dims = x_int_md.get_dims();

    dnnl::memory::desc x_f32_md({dims[0], dims[1]}, dt::f32, {dims[1], 1});
    dnnl::memory X_f32_m(x_f32_md, eng, (void *)X_f32);

    dnnl::primitive_attr q10n_attr;
    q10n_attr.set_scales_mask(DNNL_ARG_DST, /* mask */ 0);
    q10n_attr.set_zero_points_mask(DNNL_ARG_DST, /* mask */ 0);

    dnnl::reorder::primitive_desc q10n_pd(eng, x_f32_md, eng, x_int_md, q10n_attr);
    dnnl::memory dst_scale_X_m({{1}, dt::f32, {1}}, eng, &scale_X);
    dnnl::memory zp_X_m({{1}, dt::s32, {1}}, eng, &zp_X);
    dnnl::reorder(q10n_pd).execute(s,
            {{DNNL_ARG_SRC, X_f32_m}, {DNNL_ARG_DST, X_int_m},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scale_X_m},
                    {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, zp_X_m}});

    s.wait();
}

void print_f32(float* X_f32, size_t row, size_t col) {
    for (size_t i = 0; i < row*col; i++) {
        printf("%.6f ", X_f32[i]);
    }
    printf("\n");
}

void print_s8(int8_t* X_s8, size_t row, size_t col) {
    for (size_t i = 0; i < row*col; i++) {
        printf("%d ", X_s8[i]);
    }
    printf("\n");
}

int main() {
    const int64_t M = 10, N = 20, K = 30;
    dnnl::stream s(eng);
    const float threshold_dynamic_q10n = 3 * 1e-2f;   

    // Data distribution for matrices A and B
    // const float param_A_min_val = -2.f;
    // const float param_A_max_val = 1.4f;

    const float param_int8_min_val = -1.f;
    const float param_int8_max_val = -param_int8_min_val; // B is centered around 0    

    float scale_A;
    int32_t zp_A;

    size_t size_a_f32 = M * K;
    float* A_f32 = (float*)malloc(size_a_f32*sizeof(float));
    init_vector(A_f32, param_int8_min_val, param_int8_max_val, size_a_f32);
    // print_f32(A_f32, M, N);

    // We compute q10n parameters here, but in the real world applications for
    // inputs these parameters are transferred from the previous layers
    compute_q10n_params<int8_t>("A", A_f32, size_a_f32, scale_A, zp_A);
    assert(zp_A == 0 && "for int8 q10n we assume zero point = 0");

    std::vector<int8_t> A_u8(M * K, 0);
    dnnl::memory::desc a_u8_md({M, K}, dnnl::memory::data_type::s8, {K, 1});
    dnnl::memory A_u8_m(a_u8_md, eng, (void *)A_u8.data());
    quantize(A_f32, scale_A, zp_A, A_u8_m);
    // print_s8(A_u8.data(), M, N);



    float scale_B;
    int32_t zp_B;

    size_t size_b_f32 = K * N;
    float* B_f32 = (float*)malloc(size_a_f32*sizeof(float));
    init_vector(B_f32, param_int8_min_val, param_int8_max_val, size_b_f32);
    // print_f32(B_f32, M, N);

    // We compute q10n parameters here, but in the real world applications for
    // inputs these parameters are transferred from the previous layers
    compute_q10n_params<int8_t>("B", B_f32, size_b_f32, scale_B, zp_B);
    assert(zp_B == 0 && "for int8 q10n we assume zero point = 0");

    std::vector<int8_t> B_u8(M * K, 0);
    dnnl::memory::desc b_u8_md({M, K}, dnnl::memory::data_type::s8, {K, 1});
    dnnl::memory B_u8_m(b_u8_md, eng, (void *)B_u8.data());
    quantize(B_f32, scale_B, zp_B, B_u8_m);  
}
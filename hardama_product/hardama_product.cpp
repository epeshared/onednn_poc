#include <stdio.h>
#include <immintrin.h>
#include <random>
#include "oneapi/dnnl/dnnl.hpp"

// 定义矩阵大小
#define M 300
#define N 400

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

void get_filtered_matrix_onednn(float* a, int8_t* b, float* d) {
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream stream(engine);

    dnnl::memory::desc md_a({M, N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    dnnl::memory::desc md_b({M, N}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::ab);
    dnnl::memory::desc md_c({M, N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
    dnnl::memory::desc dst_md({1,1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

    dnnl::memory a_m(md_a, engine, (void *)a);
    dnnl::memory b_m(md_b, engine, (void *)b);
    dnnl::memory c_m(md_c, engine);
    dnnl::memory dst_m(dst_md, engine, d);

    dnnl::binary::primitive_desc binary_pd = 
            dnnl::binary::primitive_desc(engine, dnnl::algorithm::binary_mul, md_a, md_b, md_c);

    dnnl::reduction::primitive_desc reduce_pd =
            dnnl::reduction::primitive_desc(engine, dnnl::algorithm::reduction_sum, md_c, dst_md, 0.f, 0.f);

    auto binary_prim = dnnl::binary(binary_pd);
    auto reduction_prim = dnnl::reduction(reduce_pd);

    binary_prim.execute(stream, {{DNNL_ARG_SRC_0, a_m}, {DNNL_ARG_SRC_1, b_m}, {DNNL_ARG_DST, c_m}});
    reduction_prim.execute(stream, {{DNNL_ARG_SRC, c_m}, {DNNL_ARG_DST, dst_m}});

    stream.wait();
}


// 逐元素的乘法，只保留 b 矩阵为 1 的位置的元素
void get_filtered_matrix(float a[M][N], int8_t b[M][N], float c[M][N]) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // 如果 b 矩阵中对应位置为 1，则保留 a 矩阵中的值，否则为 0
            c[i][j] = a[i][j] * b[i][j];
        }
    }
}

// 打印矩阵
void print_matrix(const float matrix[M][N]) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%0.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

// 打印矩阵
void print_matrix_s8(const int8_t matrix[M][N]) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void init_float_mat(float* m, size_t size) {
    for (int i = 0; i < size; i++)
        m[i] = dis(gen);
}

int main() {
    float a[M][N];
    init_float_mat(&a[0][0], M*N);

    int8_t b[M][N];
    srand(time(NULL));
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            b[i][j] = rand() % 2; // 随机 0 或 1
        }
    }

    float c[M][N];

    // 获取过滤后的矩阵 c
    get_filtered_matrix(a, b, c);

    printf("Matrix A:\n");
    print_matrix(a);

    printf("\nMatrix B:\n");
    print_matrix_s8(b);

    printf("\nFiltered Matrix C:\n");
    print_matrix(c);

    float sum = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            sum += c[i][j];
        }
    }
    printf("\nSum of matrix c is: %0.2f\n", sum);    

    float d = 1.0f;
    get_filtered_matrix_onednn(&a[0][0], &b[0][0], &d);
    printf("\nFiltered Matrix d:\n");
    printf("Sum of matrix d is: %0.2f\n", d);  

    return 0;
}

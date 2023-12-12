#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <random>
#include "oneapi/dnnl/dnnl.hpp"

// Define matrix size
#define M 3
#define N 4


// // Function to calculate the sum of values in a row of matrix a based on indices in matrix b
// float calculate_sum_for_row(float* a, const uint8_t* b_row, int row_index) {
//     float sum = 0.0;
//     float* tab = a;
//     for (int i = 0; i < 3; ++i) {
//         int col_index = b_row[i]; // Assuming indices in b are 1-based
//         printf("col %d, data: %0.2f\n", col_index, tab[col_index]);
//         sum += tab[col_index];
//         tab = tab + N;
//     }
//     return sum;
// }

// void print_matrix(const float matrix[N][M]) {
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < N; ++j) {
//             printf("%0.2f ", matrix[i][j]);
//         }
//         printf("\n");
//     }
// }

// // 打印矩阵
// void print_matrix_s8(const uint8_t matrix[M][3]) {
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < 3; ++j) {
//             printf("%d ", matrix[i][j]);
//         }
//         printf("\n");
//     }
// }

// int main() {
//     // Example matrices
//     // float a[N][M]{};
//     // float* a_array = &a[0][0];
//     // for (int i = 0;i<M*N;i++) {
//     //     a_array[i] = i;
//     // }
//     float a[N][M] = {{1.0, 5.0, 9.0},
//                  {2.0, 6.0, 10.0},
//                  {3.0, 7.0, 11.0},
//                  {4.0, 8.0, 12.0}};
//     print_matrix(a);

//     uint8_t b[M][3] = {{2 - 1, 1 - 1, 4 - 1},
//                        {3 -1, 4 - 1, 2 - 1},
//                        {1 - 1, 3 - 1, 4 - 1}};

//     print_matrix_s8(b);

//     // Calculate the sum for each row in matrix b
//     for (int i = 0; i < M; ++i) {
//         float sum = calculate_sum_for_row(&a[0][0], b[i], i);

//         // Print the result for each row in matrix b
//         printf("Sum for row %d in matrix b: %f\n", i + 1, sum);
//     }

//     // Calculate the sum for each row in matrix b
//     // for (int i = 0; i < M; ++i) {
//     //     float sum = calculate_sum_for_row_oneDNN(&a[0][0], b[i], i);

//     //     // Print the result for each row in matrix b
//     //     printf("Sum for row %d in matrix b: %f\n", i + 1, sum);
//     // }    

//     return 0;
// }


/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/


#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

void resampling_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim N = 3, // batch size
            IC = 3, // channels
            IH = 227, // input tensor height
            IW = 227, // input tensor width
            OH = 350, // output tensor height
            OW = 350; // output tensor width

    // Source (src) and destination (dst) dimensions.
    memory::dims src_dims = {N, IC, IH, IW};
    memory::dims dst_dims = {N, IC, OH, OW};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> dst_data(product(dst_dims));

    // Initialize src tensor.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    // Create memory descriptors and memory objects for src and dst.
    auto src_md = memory::desc(src_dims, dt::f32, tag::nchw);
    auto dst_md = memory::desc(dst_dims, dt::f32, tag::nchw);

    auto src_mem = memory(src_md, engine);
    auto dst_mem = memory(dst_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src_data.data(), src_mem);

    // Create primitive descriptor.
    auto resampling_pd = resampling_forward::primitive_desc(engine,
            prop_kind::forward_training, algorithm::resampling_linear, src_md,
            dst_md);

    // Create the primitive.
    auto resampling_prim = resampling_forward(resampling_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> resampling_args;
    resampling_args.insert({DNNL_ARG_SRC, src_mem});
    resampling_args.insert({DNNL_ARG_DST, dst_mem});

    // Primitive execution: resampling.
    resampling_prim.execute(engine_stream, resampling_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), dst_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            resampling_example, parse_engine_kind(argc, argv));
}
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <immintrin.h>
#include <chrono>
#include <iomanip> // Include this header for std::setprecision
#define TOL 1e-4

float* a;
float* b;
float* c;
int N;
int block_size; // Added block size parameter

// Declare MatMul_AVX function
void MatMul_AVX(float* a, float* b, float* c, int srA, int scA, int srB, int scB, int srC, int scC, int dim);

void printMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::fixed << std::setprecision(2) << mat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Recursive_MatMul(float* a, float* b, float* c, int srA, int scA, int srB, int scB, int srC, int scC, int dim) {
    int newDim = dim / 2;

    if (dim <= block_size) {
        MatMul_AVX(a, b, c, srA, scA, srB, scB, srC, scC, dim); // Use AVX for the base case
        return;
    }

    Recursive_MatMul(a, b, c, srA, scA, srB, scB, srC, scC, newDim);
    Recursive_MatMul(a, b, c, srA, scA + newDim, srB + newDim, scB, srC, scC, newDim);
    Recursive_MatMul(a, b, c, srA, scA, srB, scB + newDim, srC, scC + newDim, newDim);
    Recursive_MatMul(a, b, c, srA, scA + newDim, srB + newDim, scB + newDim, srC, scC + newDim, newDim);
    Recursive_MatMul(a, b, c, srA + newDim, scA, srB, scB, srC + newDim, scC, newDim);
    Recursive_MatMul(a, b, c, srA + newDim, scA + newDim, srB + newDim, scB, srC + newDim, scC, newDim);
    Recursive_MatMul(a, b, c, srA + newDim, scA, srB, scB + newDim, srC + newDim, scC + newDim, newDim);
    Recursive_MatMul(a, b, c, srA + newDim, scA + newDim, srB + newDim, scB + newDim, srC + newDim, scC + newDim, newDim);
}

void MatMul_AVX(float* a, float* b, float* c, int srA, int scA, int srB, int scB, int srC, int scC, int dim) {
    int i, j, k;
    __m256 va, vb, vc;
    float result[8];  // Store the result elements

    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            vc = _mm256_setzero_ps();  // Initialize the result vector to zeros
            for (k = 0; k < dim; k += 8) {
                va = _mm256_loadu_ps(&a[(i + srA) * N + scA + k]);  // Load 8 single-precision values from matrix 'a'
                vb = _mm256_loadu_ps(&b[(srB + k) * N + j + scB]);  // Load 8 single-precision values from matrix 'b'
                vc = _mm256_fmadd_ps(va, vb, vc); // Multiply and accumulate
            }
            // Extract and accumulate the result into result[]
            _mm256_storeu_ps(result, vc);
            float sum = 0.0;
            for (int idx = 0; idx < 8; idx++) {
                sum += result[idx];
            }
            // Store the accumulated result in the matrix 'c'
            c[(i + srC) * N + j + scC] += sum;
        }
    }
}

void Init() {
    a = new float[N * N];
    b = new float[N * N];
    c = new float[N * N];

    if (a == nullptr || b == nullptr || c == nullptr) {
        std::cout << "Error: can't allocate memory for matrices." << std::endl;
        exit(1);
    }

    srand(time(NULL));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = 1.0f; // Initialize with float values
            b[i * N + j] = 1.0f; // Initialize with float values
        }
    }
}

void FreeMemory() {
    delete[] a;
    delete[] b;
    delete[] c;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " N block_size" << std::endl;
        exit(1);
    }

    N = atoi(argv[1]);
    block_size = atoi(argv[2]);

    int threshold = N / 2;

    if (threshold < block_size) {
        std::cout << "Block size should be smaller than or equal to N/2." << std::endl;
        exit(1);
    }

    Init();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            c[i * N + j] = 0.0f;
        }
    }

    auto start = std::chrono::steady_clock::now();
    Recursive_MatMul(a, b, c, 0, 0, 0, 0, 0, 0, N);
    auto end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

    //std::cout << "Matrix C (Result):\n";
    //printMatrix(c, N, N);

    std::cout << "Elapsed time for recursive_matmul execution = " << total_time << " seconds." << std::endl;

    FreeMemory();

    return 0;
}


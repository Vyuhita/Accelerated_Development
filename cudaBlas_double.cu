#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Function to initialize a matrix with random values
void initializeMatrixDouble(double* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i * size + j] = 1.0; // Assuming initialization with 1.0 for simplicity
        }
    }
}

// Function to print a matrix (for debugging purposes)
void printMatrixDouble(const double* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Matrix size (assuming square matrices)
    const int matrixSize = 16384;  // Adjust the size based on your requirements

    // Allocate host memory for matrices
    double* h_A = new double[matrixSize * matrixSize];
    double* h_B = new double[matrixSize * matrixSize];
    double* h_C = new double[matrixSize * matrixSize];

    // Initialize matrices with random values
    initializeMatrixDouble(h_A, matrixSize);
    initializeMatrixDouble(h_B, matrixSize);

    // Allocate device memory for matrices
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize * matrixSize * sizeof(double));
    cudaMalloc((void**)&d_B, matrixSize * matrixSize * sizeof(double));
    cudaMalloc((void**)&d_C, matrixSize * matrixSize * sizeof(double));

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, matrixSize * matrixSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize * matrixSize * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start the timer for matrix multiplication
    cudaEventRecord(start);

    // Perform matrix multiplication using cuBLAS
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrixSize, matrixSize, matrixSize, &alpha, d_A, matrixSize, d_B, matrixSize, &beta, d_C, matrixSize);

    // Stop the timer for matrix multiplication
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed time for matrix multiplication: " << elapsedTime / 1000.0 << " seconds" << std::endl;

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, matrixSize * matrixSize * sizeof(double), cudaMemcpyDeviceToHost);

    // Print the result matrix if needed
    printMatrixDouble(h_C, matrixSize);

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}


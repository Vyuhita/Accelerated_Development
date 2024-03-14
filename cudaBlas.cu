#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Function to initialize a matrix with random values
void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            //matrix[i * size + j] = static_cast<float>(rand()) / RAND_MAX;;
              matrix[i * size + j] =1.0f;
	}
    }
}

// Function to print a matrix (for debugging purposes)
void printMatrix(const float* matrix, int size) {
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
    float* h_A = new float[matrixSize * matrixSize];
    float* h_B = new float[matrixSize * matrixSize];
    float* h_C = new float[matrixSize * matrixSize];

    // Initialize matrices with random values
    initializeMatrix(h_A, matrixSize);
    initializeMatrix(h_B, matrixSize);

    // Allocate device memory for matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize * matrixSize * sizeof(float));
    cudaMalloc((void**)&d_B, matrixSize * matrixSize * sizeof(float));
    cudaMalloc((void**)&d_C, matrixSize * matrixSize * sizeof(float));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start the timer for memory copy (host to device)
    cudaEventRecord(start, 0);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);

    // Stop the timer for memory copy (host to device)
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    
    float elapsedTimeMemCopyH2D;
    cudaEventElapsedTime(&elapsedTimeMemCopyH2D, start, stop);
    std::cout << "Memory Copy (Host to Device) Time: " << elapsedTimeMemCopyH2D / 1000.0 << " seconds" << std::endl;

    // Initialize cuBLAS
    //they are used to encapsulate and manage resources efficiently.
   // They abstract away the internal details of the library and provide a clean interface for your application 
    //to interact with the GPU-related functionality.
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Start the timer for matrix multiplication
    cudaEventRecord(start, 0);

    // Perform matrix multiplication using cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrixSize, matrixSize, matrixSize, &alpha, d_A, matrixSize, d_B, matrixSize, &beta, d_C, matrixSize);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time for matrix multiplication
    float elapsedTimeMultiplication;
    cudaEventElapsedTime(&elapsedTimeMultiplication, start, stop);
    std::cout << "Matrix Multiplication Time (including Memory Copy): " << elapsedTimeMultiplication / 1000.0 << " seconds" << std::endl;

    
    cudaEventRecord(start, 0);

    
    cudaMemcpy(h_C, d_C, matrixSize * matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    
    float elapsedTimeMemCopyD2H;
    cudaEventElapsedTime(&elapsedTimeMemCopyD2H, start, stop);
    std::cout << "Memory Copy (Device to Host) Time: " << elapsedTimeMemCopyD2H / 1000.0 << " seconds" << std::endl;
    std::cout << "Total Time: " << elapsedTimeMemCopyH2D / 1000.0+ elapsedTimeMemCopyD2H / 1000.0+ elapsedTimeMultiplication / 1000.0<< " seconds" << std::endl;
   // printMatrix(h_C, matrixSize);
    
    
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}


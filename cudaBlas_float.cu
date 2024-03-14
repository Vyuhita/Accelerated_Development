#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i * size + j] = 1.0f;
        }
    }
}

void printMatrix(const float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    const int matrixSize = 4096;

    float* h_A = new float[matrixSize * matrixSize];
    float* h_B = new float[matrixSize * matrixSize];
    float* h_C = new float[matrixSize * matrixSize];

    initializeMatrix(h_A, matrixSize);
    initializeMatrix(h_B, matrixSize);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize * matrixSize * sizeof(float));
    cudaMalloc((void**)&d_B, matrixSize * matrixSize * sizeof(float));
    cudaMalloc((void**)&d_C, matrixSize * matrixSize * sizeof(float));

    //cudaMemcpy(d_A, h_A, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_B, h_B, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start the timer for memory copy (host to device)
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start); // Synchronize the event recording
    
    cudaMemcpy(d_A, h_A, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrixSize, matrixSize, matrixSize, 
                &alpha, d_A, matrixSize, d_B, matrixSize, &beta, d_C, matrixSize);

    // Stop the timer for memory copy (host to device)
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // Synchronize the event recording

    // Calculate and print the elapsed time for memory copy (host to device)
    float elapsedTimeMemCopyH2D;
    cudaEventElapsedTime(&elapsedTimeMemCopyH2D, start, stop);
    std::cout << "Memory Copy (Host to Device) Time: " << elapsedTimeMemCopyH2D / 1000.0 << " seconds" << std::endl;

    // Start the timer for memory copy (device to host)
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start); // Synchronize the event recording

    cudaMemcpy(h_C, d_C, matrixSize * matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Stop the timer for memory copy (device to host)
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // Synchronize the event recording

    // Calculate and print the elapsed time for memory copy (device to host)
    float elapsedTimeMemCopyD2H;
    cudaEventElapsedTime(&elapsedTimeMemCopyD2H, start, stop);
    std::cout << "Memory Copy (Device to Host) Time: " << elapsedTimeMemCopyD2H / 1000.0 << " seconds" << std::endl;

    // Calculate and print the elapsed time for matrix multiplication
    float elapsedTimeMultiplication = elapsedTimeMemCopyH2D + elapsedTimeMemCopyD2H;
    std::cout << "Matrix Multiplication Time (including Memory Copy): " << elapsedTimeMultiplication / 1000.0 << " seconds" << std::endl;
    printMatrix(h_C, matrixSize);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}


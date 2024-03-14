#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 8192 
#define BLOCK_SIZE 32
#define NUM_GPUS 2

// Function to print a matrix
void printMatrix(int* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Define the timer function
float timer(cudaEvent_t start, cudaEvent_t stop) {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    return elapsedTime;
}

__global__ void Kernel_A(int* a, int* b, int* c, int matrixSize) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    __shared__ int shared_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int shared_b[BLOCK_SIZE][BLOCK_SIZE];

    int sum = 0;

    for (int m = 0; m < matrixSize / BLOCK_SIZE; ++m) {
        shared_a[ty][tx] = a[row * matrixSize + (m * BLOCK_SIZE + tx)];
        shared_b[ty][tx] = b[col + (m * BLOCK_SIZE + ty) * matrixSize];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += shared_a[ty][k] * shared_b[k][tx];
        }
        __syncthreads();
    }

    c[row * matrixSize + col] = sum;
}

void matrixMultiplication(int device, int* h_a, int* h_b, int* h_c, int numRows) {
    int* d_a;
    int* d_b;
    int* d_c;

    int size = N * numRows * sizeof(int);

    //device memory allocation
    cudaSetDevice(device);
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Create CUDA streams
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Record start time
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    // Copy data from host to device using streams
    cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (numRows + blockDim.y - 1) / blockDim.y);

    // Launch Kernel_A
    Kernel_A<<<gridDim, blockDim, 0, stream>>>(d_a, d_b, d_c, N);

    // Synchronize the stream
    cudaStreamSynchronize(stream);

    // Copy result from device to host
    cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream);

    // Record end time
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    // Calculate elapsed time including memory transfers and kernel execution
    float elapsedTime = timer(startEvent, stopEvent);
    std::cout << "Elapsed Time on GPU " << device << ": " << elapsedTime << " ms" << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Destroy CUDA stream
    cudaStreamDestroy(stream);
}

int main() {
    // Initialize two GPUs
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    if (numGPUs < NUM_GPUS) {
        std::cerr << "This code requires at least " << NUM_GPUS << " GPUs." << std::endl;
        return 1;
    }

    float gpuTimes[NUM_GPUS] = {0.0};

    // Create data matrices
    int* h_a = (int*)malloc(N * N * sizeof(int));
    int* h_b = (int*)malloc(N * N * sizeof(int));
    int* h_c0 = (int*)malloc(N * N * sizeof(int));
    int* h_c1 = (int*)malloc(N * N * sizeof(int));

    // Initialize input matrices with random values
    for (int i = 0; i < N * N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 1.0f;
    }

    // Define the number of rows for each GPU
    int numRowsGPU0 = N / 2;
    int numRowsGPU1 = N - numRowsGPU0;

    // Start the timer for the entire process
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start, 0);

    // Launch matrix multiplication on GPU 0 for the first numRowsGPU0 rows
    matrixMultiplication(0, h_a, h_b, h_c0, numRowsGPU0);

    // Record end time for GPU 0
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    gpuTimes[0] = elapsedTime;

    // Reset the timer for GPU 1
    cudaEventRecord(start, 0);

    // Launch matrix multiplication on GPU 1 for the remaining numRowsGPU1 rows
    matrixMultiplication(1, h_a + numRowsGPU0 * N, h_b, h_c1, numRowsGPU1);

    // Record end time for GPU 1
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    gpuTimes[1] = elapsedTime;

    // Calculate the total elapsed time by summing the GPU times
    float totalElapsedTime = gpuTimes[0] + gpuTimes[1];
    std::cout << "Total Elapsed Time: " << totalElapsedTime / 1000.0 << " seconds" << std::endl;

    // Combine results if needed (h_c0 and h_c1 now contain partial results)
	// Combine h_c0 and h_c1 into a single result matrix h_c
int* h_c = (int*)malloc(N * N * sizeof(int));

// Copy h_c0 to the upper part of h_c
for (int i = 0; i < numRowsGPU0; ++i) {
    memcpy(h_c + i * N, h_c0 + i * N, N * sizeof(int));
}

// Copy h_c1 to the lower part of h_c
for (int i = 0; i < numRowsGPU1; ++i) {
    memcpy(h_c + (i + numRowsGPU0) * N, h_c1 + i * N, N * sizeof(int));
}

// Now, h_c contains the combined result
	printMatrix(h_c, N);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}


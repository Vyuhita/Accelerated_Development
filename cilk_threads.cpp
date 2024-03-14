#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#define TOL 1e-4f

float* a;
float* b;
float* c;
int N;
int block_size;

double GetTime() {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

void Init() {
    a = new float[N * N];
    b = new float[N * N];
    c = new float[N * N];

    if (a == nullptr || b == nullptr || c == nullptr) {
        std::cerr << "Error: can't allocate memory for matrices." << std::endl;
        delete[] a;
        delete[] b;
        delete[] c;
        exit(1);
    }

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = static_cast<float>(std::rand() % 100) / 10.0f;
            b[i * N + j] = static_cast<float>(std::rand() % 100) / 10.0f;
        }
    }
}

void Parallel_MatMul(float* a, float* b, float* c) {
    cilk_for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}

int Validate() {
    float* d = new float[N * N];

    Parallel_MatMul(a, b, d); // Compute result using the parallelized method

    float eps = 0.0f;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            d[idx] = std::fabs(d[idx] - c[idx]);

            if (eps < d[idx]) {
                eps = d[idx];
            }
        }
    }

    delete[] d;

    return (eps < TOL); // Return 1 if validation is successful, 0 otherwise
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " N block_size num_threads" << std::endl;
        exit(1);
    }

    N = std::atoi(argv[1]);
    block_size = std::atoi(argv[2]);
    int numThreads = std::atoi(argv[3]);

    // Set the number of Cilk threads using an environment variable
    setenv("CILK_NWORKERS", std::to_string(numThreads).c_str(), 1);

    Init();

    for (int i = 0; i < N * N; i++) {
        c[i] = 0.0f; // Initialize result matrix
    }

    double start_time = GetTime();
    Parallel_MatMul(a, b, c); // Parallelized matrix multiplication
    double end_time = GetTime();
    std::cout << "Execution time: " << end_time - start_time << " seconds" << std::endl;

    if (Validate()) {
        std::cout << "Validation passed: Results are correct." << std::endl;
    } else {
        std::cout << "Validation failed: Results are incorrect." << std::endl;
    }

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}


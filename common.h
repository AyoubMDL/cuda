#include <ctime>
#include <cstdlib>
#include <chrono>

void initializeVectors(float *A, float *B, int N) {
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}


template <typename Func>
double measureExecutionTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}


bool compareResults(const float *A, const float *B, int N) {
    for (int i = 0; i < N; i++) {
        if (fabs(A[i] - B[i]) > 1e-5) {
            std::cout << "Mismatch at index " << i << ": CPU=" << A[i] << " GPU=" << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

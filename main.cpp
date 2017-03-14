#include <iostream>
#include <iomanip>
#include <cmath>
#include <pthread.h>
#include <cstdlib>
#include <chrono>

pthread_barrier_t barrier;
pthread_mutex_t mutex;

struct thread_info {
    double **A;
    double *B;
    int size;
    int threads_count;
    int thread_id;
    int *indexOfMax;
};

void gaussianElimination(double **A, double *B, int size);

void gaussianEliminationParallel(double **A, double *B, int size, int threads_count, int thread_id, int *indexesOfMax);

void backSubstitution(double **A, double *B, int size);

void backSubstitutionParallel(double **A, double *B, int size, int threads_count, int thread_id);

void printLinearSystem(double **A, double *B, int size);

double **createLinearSystemCoefficients(int size);

double *createLinearSystemConstantTerms(int size);

void deleteMatrixOfDouble(double **matrix, int size);

void deleteVectorOfDouble(double *vector);

void *thread_func(void *args);

int main(int argc, char **argv) {

    if (argc < 3) {
        std::cout << "Not enough args" << std::endl;
        return -1;
    }

    int size = atoi(argv[1]);
    int threads_count = atoi(argv[2]);

    std::cout << "size: " << size << std::endl;
    std::cout << "threads: " << threads_count << std::endl;

    srand((unsigned int) time(0));

    double **matrix = createLinearSystemCoefficients(size);
    double *vector = createLinearSystemConstantTerms(size);

    if (size <= 10) {
        printLinearSystem(matrix, vector, size);
    }

    auto get_time = std::chrono::steady_clock::now;
    decltype(get_time()) start, end;
    start = get_time();

    int *indexOfMax = new int[threads_count];
    pthread_t threads[threads_count];
    thread_info info[threads_count];

    pthread_barrier_init(&barrier, NULL, (unsigned int) threads_count);
    pthread_mutex_init(&mutex, NULL);

    for (int i = 0; i < threads_count; ++i) {
        info[i].threads_count = threads_count;
        info[i].thread_id = i;
        info[i].A = matrix;
        info[i].B = vector;
        info[i].size = size;
        info[i].indexOfMax = indexOfMax;
        pthread_create(&threads[i], NULL, thread_func, &info[i]);
    }

    for (int i = 0; i < threads_count; ++i) {
        pthread_join(threads[i], NULL);
    }

    end = get_time();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (size <= 10) {
        printLinearSystem(matrix, vector, size);
    }

    std::cout << "Result time: " << double(elapsed) / 1000.0 << " s.\n";

    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&mutex);

    deleteVectorOfDouble(vector);
    deleteMatrixOfDouble(matrix, size);

    return 0;
}

void *thread_func(void *args) {
    thread_info *info = (thread_info *) args;
    gaussianEliminationParallel(info->A, info->B, info->size, info->threads_count, info->thread_id, info->indexOfMax);
    backSubstitutionParallel(info->A, info->B, info->size, info->threads_count, info->thread_id);
    return 0;
}

void gaussianElimination(double **A, double *B, int size) {
    for (int i = 0; i < size; ++i) {

        int indexOfMax = i;
        for (int l = i + 1; l < size; ++l) {
            if (fabs(A[l][i]) >= fabs(A[indexOfMax][i])) {
                indexOfMax = l;
            }
        }

        std::swap(A[i], A[indexOfMax]);
        std::swap(B[i], B[indexOfMax]);

        for (int k = i + 1; k < size; ++k) {

            double koef = A[k][i] / A[i][i];
            B[k] -= koef * B[i];

            for (int j = i; j < size; ++j) {
                A[k][j] -= koef * A[i][j];
            }
        }
    }
}


void gaussianEliminationParallel(double **A, double *B, int size, int threads_count, int thread_id, int *indexesOfMax) {
    for (int i = 0; i < size; ++i) {

        int indexOfMaxInCurrentThread = i;
        for (int l = i + 1 + thread_id; l < size; l += threads_count) {
            if (fabs(A[l][i]) >= fabs(A[indexOfMaxInCurrentThread][i])) {
                indexOfMaxInCurrentThread = l;
            }
        }
        indexesOfMax[thread_id] = indexOfMaxInCurrentThread;

        if (pthread_barrier_wait(&barrier) == PTHREAD_BARRIER_SERIAL_THREAD) {
            int indexOfMax = indexesOfMax[0];
            for (int j = 1; j < threads_count; ++j) {
                if (fabs(A[indexesOfMax[j]][i]) >= fabs(A[indexOfMax][i])) {
                    indexOfMax = indexesOfMax[j];
                }
            }

            std::swap(A[i], A[indexOfMax]);
            std::swap(B[i], B[indexOfMax]);
        }

        pthread_barrier_wait(&barrier);

        for (int k = i + 1 + thread_id; k < size; k += threads_count) {
            double koef = A[k][i] / A[i][i];
            B[k] -= koef * B[i];

            for (int j = i; j < size; ++j) {
                A[k][j] -= koef * A[i][j];
            }
        }

        pthread_barrier_wait(&barrier);
    }
}


void backSubstitution(double **A, double *B, int size) {
    for (int i = size - 1; i >= 0; --i) {
        double sum = 0.0;
        double koef;
        for (int j = i + 1; j < size; ++j) {
            koef = A[i][j] / A[j][j];
            A[i][j] -= koef * A[j][j];
            sum += koef * B[j];
        }

        B[i] = (B[i] - sum) / A[i][i];
        A[i][i] /= A[i][i];
    }
}

void backSubstitutionParallel(double **A, double *B, int size, int threads_count, int thread_id) {

    for (int i = size - 1; i >= 0; --i) {
        double sum = 0.0;
        double koef;
        for (int j = i + 1 + thread_id; j < size; j += threads_count) {
            koef = A[i][j] / A[j][j];
            A[i][j] -= koef * A[j][j];
            sum += koef * B[j];
        }

        pthread_mutex_lock(&mutex);
        B[i] = B[i] - sum;
        pthread_mutex_unlock(&mutex);


        if (pthread_barrier_wait(&barrier) == PTHREAD_BARRIER_SERIAL_THREAD) {
            B[i] /= A[i][i];
            A[i][i] /= A[i][i];
        }

        pthread_barrier_wait(&barrier);
    }
}

void printLinearSystem(double **A, double *B, int size) {
    std::cout.precision(3);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << "|" << std::setw(10) << A[i][j] << "  ";
        }
        std::cout << "|| " << B[i] << std::endl;
    }
    std::cout << std::endl;
}

double **createLinearSystemCoefficients(int size) {
    double **result = new double *[size];
    for (int i = 0; i < size; ++i) {
        result[i] = new double[size];
        for (int j = 0; j < size; ++j) {
            result[i][j] = (double) (rand() % 10);
        }
    }

    return result;
}

double *createLinearSystemConstantTerms(int size) {
    double *result = new double[size];
    for (int i = 0; i < size; ++i) {
        result[i] = (double) (rand() % 10);
    }

    return result;
}

void deleteMatrixOfDouble(double **matrix, int size) {
    for (int i = 0; i < size; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void deleteVectorOfDouble(double *vector) {
    delete[] vector;
}



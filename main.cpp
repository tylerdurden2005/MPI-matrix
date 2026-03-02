#include <iostream>
#include <vector>
#include "mpi/mpi.h"
#include <ctime>

using namespace std;

void initMatrix(vector<int>& A, vector<int>& B, int n){
    srand(time(0));
    for (int i = 0; i < n * n; i++) {
        A[i] = rand() % 10;
    }
    for (int i = 0; i < n * n; i++) {
        B[i] = rand() % 10;
    }
}


void classicMatrixMult(vector<int>& A, vector<int>& B, vector<int>& C, int n){
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; ++k) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void printAnswer(vector<int>& C, int n){
    int max_display = 5;
    if (n > max_display) {
        cout << "Part of matrix:" << endl;
        for (int i = 0; i < max_display; ++i){
            for (int j = 0; j < max_display; ++j){
                cout << C[i * n + j] << " ";
            }
            cout << "..." << endl;
        }
        cout << "..." << endl;
    } else {
        for (int i = 0; i < n; ++i){
            for (int j = 0; j < n; ++j){
                cout << C[i * n + j] << " ";
            }
            cout << endl;
        }
    }
}

int main(int argc, char* argv[]){
    
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int m = 100;
    vector<int> A, B, C;

    if (rank==0){
        A.resize(m * m);
        B.resize(m * m);
        C.resize(m * m);
        initMatrix(A, B, m);
    }
    int k = m / size;

    vector<int> local_A(k * m);
    vector<int> local_B(m * k);
    vector<int> local_C(k * m, 0);

    vector<int> stripedB; 

    if (rank == 0) {
        stripedB.resize(m * m);
        for (int block = 0; block < size; ++block) {
            for (int col_in_block = 0; col_in_block < k; ++col_in_block) {
                for (int row = 0; row < m; ++row) {
                    int global_col = block * k + col_in_block;
                    stripedB[block * (m * k) + col_in_block * m + row] = B[row * m + global_col];
                }
            }
        }
    }
    
    MPI_Scatter(A.data(), k * m, MPI_INT,
                local_A.data(), k * m, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Scatter(stripedB.data(), m * k, MPI_INT,  
                local_B.data(), m * k, MPI_INT,   
                0, MPI_COMM_WORLD);

    int dims[1] = {size};
    int periods[1] = {1};
    int reorder = 0;
    MPI_Comm ring_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &ring_comm);

    int prev_rank, next_rank;
    MPI_Cart_shift(ring_comm, 0, 1, &prev_rank, &next_rank);
    
     for (int iteration = 0; iteration < size; ++iteration) {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                int sum = 0;
                for (int kk = 0; kk < m; ++kk) {
                    sum += local_A[i * m + kk] * local_B[j * m + kk];
                }
                int global_block = (rank - iteration + size) % size;
                int global_col = global_block * k + j;
                local_C[i * m + global_col] += sum;
            }
        }
        if (iteration < size - 1) {
            MPI_Sendrecv_replace(local_B.data(), m * k, MPI_INT,
                                 next_rank, 0, 
                                 prev_rank, 0, 
                                 ring_comm, MPI_STATUS_IGNORE);
        }
    }
    

    MPI_Gather(local_C.data(), k * m, MPI_INT, C.data(), k * m, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank==0){
       	printAnswer(C, m);
        vector<int> R(m*m);
        classicMatrixMult(A, B, R, m);
        if (R == C) cout << "The answers matched!" << endl;
        //printAnswer(R, m);
    }
    MPI_Comm_free(&ring_comm);
    MPI_Finalize();
    return 0;
}




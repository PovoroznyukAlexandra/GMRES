#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include "mpi.h"

using namespace std;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution <>floatDist(-1, 0);


// Функция для умножения матрицы на вектор
void MatVec(double** A, int n, const double* x, int m, double *res) {
    int my_rank, total_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

    int extra, offset;
    int dest, source;
    extra = my_rank < n % total_processes ? 0 : 1;
    offset = my_rank * m;
    if (extra)
        offset += m % total_processes;
    dest = (my_rank + total_processes - 1) % total_processes;
    source = (my_rank + 1) % total_processes;
    auto *v = new double[n + extra];
    auto *A_loc = new double[n*n];
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            A_loc[i] = A[i][j];
        }
    }



    memcpy(v, x, sizeof(double) * (n + extra));
    std::fill(res, res + n, 0.0);
    for (int k = 0; k < total_processes; ++k) {
        int col_block = m / total_processes;
        if ((k + my_rank) % total_processes < m % total_processes) {
            ++col_block;
        }
        MPI_Request requests[2];
        MPI_Status statuses[2];
        // send our data to next process and recieve new data from previous process
        MPI_Irecv(x, n + extra, MPI_DOUBLE, source, 21, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(v, n + extra, MPI_DOUBLE, dest, 21, MPI_COMM_WORLD, &requests[1]);
        // while getting new data - perform work on current data
        for (int i = 0; i < n; ++i) {
            res[i] = 0;
            for (int j = 0; j < col_block; ++j) {
                res[i] += A_loc[i * m + (j + offset)] * v[j];
            }
        }
        offset = (offset + col_block) % m;
        // wait for getting new data
        MPI_Waitall(2, requests, statuses);
        memcpy(v, x, sizeof(double) * (n + extra));
    }
}

/*
double** Transpose(double** Q, int n, int m){
    auto **Q_T = new double* [m];
    for (int i = 0; i < m; i++){
        Q_T[i] = new double [n];
    }
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            Q_T[j][i] = Q[i][j];
        }
    }
    return Q_T;
}
 */

// Функция для вычисления нормы вектора
double Norm(const double* x, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

/*
// Функция для вычисления скалярного произведения
double Dot(const double* x, const double* y, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}
*/

double* Solve_Upper_Triangular(double** R, const double* b, int n){
    auto* x = new double [n];
    // Инициализация вектора решения нулями
    for (int i = 0; i < n; ++i){
        x[i] = 0;
    }
    // Решение системы методом обратной подстановки
    for (int i = n - 1; i >= 0; --i){
        double sum = 0;
        for (int j = i + 1; j < n; ++j){
            sum += R[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / R[i][i];
    }
    return x;
}


double* rotation(const double* vec, const vector<double>& cotangences, int n){
    assert(n > cotangences.size());
    auto *h = new double [n];
    auto* res = new double [n];
    for (int j = 0; j < n; j++){
        res[j] = vec[j];
    }
    int i = 0;
    for (double ctg : cotangences){
        double s = 1 / sqrt(pow(ctg, 2) + 1);
        double c = ctg * s;
        double temp1 = c * vec[i] + s * vec[i + 1];
        double temp2 = -s * vec[i] + c * vec[i + 1];
        res[i] = temp1;
        res[i + 1] = temp2;
        i += 1;
    }
    for (int j = 0; j < n; j++){
        h[j] = res[j];
    }
    return h;
}


// Функция GMRES
double* GMRES(double** A, const double* b, int n, int k, double eps) {
    auto **V = new double* [n];
    auto* x = new double [n];
    for (int i = 0; i < n; i++){
        V[i] = new double [k];
        for (int j = 0; j < k; j++){
            V[i][j] = 0;
        }
    }

    auto **R = new double* [k];  //H = QR
    for (int i = 0; i < k; i++){
        R[i] = new double [k];
        for (int j = 0; j < k; j++){
            R[i][j] = 0;
        }
    }

    auto* e_1_rot = new double[n];
    e_1_rot[0] = 1;
    for (int i = 1; i < n; i++){
        e_1_rot[i] = 0;
    }

    // step 0
    int m = 0;
    double beta = Norm(b, n);
    auto* v = new double[n];
    auto* res = new double[n];
    for (int i = 0; i < n; i++){
        v[i] = b[i]/beta;
        V[i][0] = v[i];
    }
    double residual = beta;
    //ctg = []
    vector <double> ctg;

    while (residual > eps){
        m += 1;   //step m
        if (m > k-1){
            cout << "Number of iterations > k" << endl;
            return x;   // return b, residual
        }
        MatVec(A, n, v ,n, res);
        for (int i = 0; i < n; i++){
            v[i] = res[i];
        }
        auto* h = new double[m+1];
        for (int i = 0; i < m; i++) {
            h[i] = 0;
            for (int j = 0; j < n; j++) {
                h[i] += V[j][i] * v[j];
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                v[i] -= V[i][j] * h[j];
            }
        }
        h[m] = Norm(v, n);
        auto* c = new double[m];
        if (h[m] < 1e-14){
            cout << "dim K = n" << endl;
            auto* c_1 = new double[m-1];
            auto* e_1_rot_1 = new double[m-1];
            auto **R_1 = new double* [m-1];
            for (int i = 0; i < m-1; i++){
                R_1[i] = new double [m-1];
                for (int j = 0; j < m-1; j++){
                    R_1[i][j] = R[i][j];
                }
                c_1[i] = c[i];
                e_1_rot_1[i] = beta * e_1_rot[i];
            }
            c = Solve_Upper_Triangular(R_1, e_1_rot_1, m-1);
            auto **V_1 = new double* [n];
            for (int i = 0; i < n; i++){
                V_1[i] = new double [m-1];
                for (int j = 0; j < m-1; j++){
                    V_1[i][j] = V[i][j];
                }
            }
            MatVec(V_1, n, c, m-1, x);
            cout << h[m] << endl;
            return x;  //return x
        }
        for (int i = 0; i < n; i++){
            v[i] /= h[m];
            V[i][m] = v[i];
        }
        h = rotation(h, ctg, m+1);
        ctg.push_back(h[m - 1]/ h[m]);

        double& ind = ctg.back();
        vector<double> ctg_1 = {ind};
        auto* tmp = new double[2];

        tmp[0] = h[m-1];
        tmp[1] = h[m];
        tmp = rotation(tmp, ctg_1, 2);
        h[m-1] = tmp[0];
        h[m] = tmp[1];
        assert(h[m] < 1e-8);
        for (int i = 0; i < m+1; i++){
            R[i][m-1] = h[i];
        }

        tmp[0] = e_1_rot[m-1];
        tmp[1] = e_1_rot[m];
        tmp = rotation(tmp, ctg_1, 2);
        e_1_rot[m-1] = tmp[0];
        e_1_rot[m] = tmp[1];

        assert(h[m] < 1e-8);
        for (int i = 0; i < m+1; i++){
            R[i][m-1] = h[i];
        }

        residual = beta * abs(e_1_rot[m]);
        auto* e_1_rot_2 = new double[m];
        auto **R_2 = new double* [m];
        for (int i = 0; i < m; i++){
            R_2[i] = new double [m];
            for (int j = 0; j < m; j++){
                R_2[i][j] = R[i][j];
            }
            e_1_rot_2[i] = beta * e_1_rot[i];
        }
        c = Solve_Upper_Triangular(R_2, e_1_rot_2, m-1);
        auto **V_2 = new double* [n];
        for (int i = 0; i < n; i++){
            V_2[i] = new double [m-1];
            for (int j = 0; j < m-1; j++){
                V_2[i][j] = V[i][j];
            }
        }
        MatVec(V_2, n, c, m, x);
    }
    cout << "m = " << m << endl;
    return x;
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    cout << "╰( ͡° ͜ʖ ͡° )つ──☆*" << endl;
    int n = 10;
    double tol = 1e-6;
    int maxit = 100;
    srand(time(nullptr));

    auto* b = new double [n];
    auto* x = new double[n];
    auto **A = new double* [n];
    for (int i = 0; i < n; i++){
        A[i] = new double [n];
        b[i] = (-floatDist(gen));
        for (int j = 0; j < n; j++){
            A[i][j] = (-floatDist(gen));
        }
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    x = GMRES(A, b, n, maxit, tol);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time for GMRES: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

    for (int i = 0; i < n; i++){
        cout << x[i] << ' ';
    }
    cout << endl;

    for (int i = 0; i < n; i++){
        delete[] A[i];
    }
    delete [] A; // Удаляем массивы
    delete [] b;
    delete [] x;

    MPI_Finalize();
    return 0;
}
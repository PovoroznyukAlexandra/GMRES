#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <cstring>
#include <cassert>
#include "mpi.h"

using namespace std;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> floatDist(-1, 0);

// Функция для умножения матрицы на вектор
double *MatVec(double *local_matrix, double *local_vector, int local_n, int m)
{
    double *local_result = new double[local_n];
    for (int i = 0; i < local_n; i++)
    {
        local_result[i] = 0.0;
        for (int j = 0; j < m; j++)
        {
            local_result[i] += local_matrix[i * m + j] * local_vector[j];
        }
    }
    return local_result;
}

double *MatVecWrap(double **A, int n, double *local_vector, int m)
{
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_n = n / size; // Число строк на каждом процессе

    if (rank == size - 1)
    {
        local_n += n % size;
    }

    int *shifts_rcv = new int[size];
    int *counts_rcv = new int[size];
    int *shifts_snd = new int[size];
    int *counts_snd = new int[size];
    double *new_matrix = new double[n * m];
    double *local_matrix = new double[local_n * m];

    if (rank == 0)
    {
        for (int i = 0; i < n; ++i)
        {
            std::memcpy(new_matrix + i * m, A[i], m * sizeof(double));
        }
    }

    for (int i = 0; i < size; ++i)
    {
        counts_rcv[i] = (i != size - 1) ? n / size : n / size + n % size;
        shifts_rcv[i] = (n / size) * i;
        counts_snd[i] = counts_rcv[i] * m;
        shifts_snd[i] = shifts_rcv[i] * m;
    }

    int error_code;

    error_code = MPI_Scatterv(new_matrix, counts_snd, shifts_snd, MPI_DOUBLE, local_matrix, local_n * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    error_code = MPI_Bcast(local_vector, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *local_result = MatVec(local_matrix, local_vector, local_n, m);

    double *result = new double[n];

    error_code = MPI_Gatherv(local_result, local_n, MPI_DOUBLE, result, counts_rcv, shifts_rcv, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(result, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[] local_result;
    delete[] local_matrix;
    delete[] shifts_rcv;
    delete[] shifts_snd;
    delete[] counts_rcv;
    delete[] counts_snd;
    delete[] new_matrix;

    return result;
}
// Функция для вычисления нормы вектора
double Norm(const double *x, int n)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += x[i] * x[i];
    }
    return sqrt(sum);
}

double *Solve_Upper_Triangular(double **R, const double *b, int n)
{
    auto *x = new double[n];
    // Инициализация вектора решения нулями
    for (int i = 0; i < n; ++i)
    {
        x[i] = 0;
    }
    // Решение системы методом обратной подстановки
    for (int i = n - 1; i >= 0; --i)
    {
        double sum = 0;
        for (int j = i + 1; j < n; ++j)
        {
            sum += R[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / R[i][i];
    }
    return x;
}

double *rotation(const double *vec, const vector<double> &cotangences, int n)
{
    assert(n > cotangences.size());
    auto *h = new double[n];
    auto *res = new double[n];
    for (int j = 0; j < n; j++)
    {
        res[j] = vec[j];
    }
    int i = 0;
    for (double ctg : cotangences)
    {
        double s = 1 / sqrt(pow(ctg, 2) + 1);
        double c = ctg * s;
        double temp1 = c * vec[i] + s * vec[i + 1];
        double temp2 = -s * vec[i] + c * vec[i + 1];
        res[i] = temp1;
        res[i + 1] = temp2;
        i += 1;
    }
    for (int j = 0; j < n; j++)
    {
        h[j] = res[j];
    }
    return h;
}

// Функция GMRES
double *GMRES(double **A, const double *b, int n, int k, double eps)
{
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double **V;
    double *x = new double[n];
    double **R; // H = QR
    double *e_1_rot;

    if (rank == 0)
    {
        V = new double *[n];
        R = new double *[k];

        for (int i = 0; i < n; i++)
        {
            V[i] = new double[k];
            for (int j = 0; j < k; j++)
            {
                V[i][j] = 0;
            }
        }
        for (int i = 0; i < k; i++)
        {
            R[i] = new double[k];
            for (int j = 0; j < k; j++)
            {
                R[i][j] = 0;
            }
        }
        e_1_rot = new double[n];
        e_1_rot[0] = 1;
        for (int i = 1; i < n; i++)
        {
            e_1_rot[i] = 0;
        }
    }

    // step 0
    int m = 0;
    double beta = Norm(b, n);

    auto *v = new double[n];

    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            v[i] = b[i] / beta;
            V[i][0] = v[i];
        }
    }

    double *residual = new double[1];
    *residual = beta;
    MPI_Bcast(residual, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // ctg = []
    vector<double> ctg;

    for (int CNT = 0; CNT < k; CNT++)
    {
        m += 1; // step m

        if (*residual < eps)
        {
            break;
        }

        v = MatVecWrap(A, n, v, n);

        double *h_m = new double[1];
        double *h;
        if (rank == 0)
        {
            h = new double[m + 1];
            for (int i = 0; i < m; i++)
            {
                h[i] = 0;
                for (int j = 0; j < n; j++)
                {
                    h[i] += V[j][i] * v[j];
                }
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    v[i] -= V[i][j] * h[j];
                }
            }
            h[m] = Norm(v, n);
            *h_m = h[m];
        }
        MPI_Bcast(h_m, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        auto *c = new double[m];
        if (*h_m < 1e-14)
        {
            double **V_1;

            if (rank == 0)
            {
                V_1 = new double *[n];
                cout << "dim K = n" << endl;
                auto *c_1 = new double[m - 1];
                auto *e_1_rot_1 = new double[m - 1];
                auto **R_1 = new double *[m - 1];
                for (int i = 0; i < m - 1; i++)
                {
                    R_1[i] = new double[m - 1];
                    for (int j = 0; j < m - 1; j++)
                    {
                        R_1[i][j] = R[i][j];
                    }
                    c_1[i] = c[i];
                    e_1_rot_1[i] = beta * e_1_rot[i];
                }
                c = Solve_Upper_Triangular(R_1, e_1_rot_1, m - 1);
                for (int i = 0; i < n; i++)
                {
                    V_1[i] = new double[m - 1];
                    for (int j = 0; j < m - 1; j++)
                    {
                        V_1[i][j] = V[i][j];
                    }
                }
            }
            x = MatVecWrap(V_1, n, c, m - 1);
            // cout << *h_m << endl;
            return x; // return x
        }

        double **V_2;

        if (rank == 0)
        {
            for (int i = 0; i < n; i++)
            {
                v[i] /= h[m];
                V[i][m] = v[i];
            }
            h = rotation(h, ctg, m + 1);
            ctg.push_back(h[m - 1] / h[m]);

            double &ind = ctg.back();
            vector<double> ctg_1 = {ind};
            auto *tmp = new double[2];

            tmp[0] = h[m - 1];
            tmp[1] = h[m];
            tmp = rotation(tmp, ctg_1, 2);
            h[m - 1] = tmp[0];
            h[m] = tmp[1];
            assert(h[m] < 1e-8);
            for (int i = 0; i < m + 1; i++)
            {
                R[i][m - 1] = h[i];
            }

            tmp[0] = e_1_rot[m - 1];
            tmp[1] = e_1_rot[m];
            tmp = rotation(tmp, ctg_1, 2);
            e_1_rot[m - 1] = tmp[0];
            e_1_rot[m] = tmp[1];

            assert(h[m] < 1e-8);
            for (int i = 0; i < m + 1; i++)
            {
                R[i][m - 1] = h[i];
            }

            *residual = beta * abs(e_1_rot[m]);

            auto *e_1_rot_2 = new double[m];
            auto **R_2 = new double *[m];
            for (int i = 0; i < m; i++)
            {
                R_2[i] = new double[m];
                for (int j = 0; j < m; j++)
                {
                    R_2[i][j] = R[i][j];
                }
                e_1_rot_2[i] = beta * e_1_rot[i];
            }
            c = Solve_Upper_Triangular(R_2, e_1_rot_2, m - 1);
            V_2 = new double *[n];
            for (int i = 0; i < n; i++)
            {
                V_2[i] = new double[m - 1];
                for (int j = 0; j < m - 1; j++)
                {
                    V_2[i][j] = V[i][j];
                }
            }
        }
        MPI_Bcast(residual, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        x = MatVecWrap(V_2, n, c, m);
    }
    if (rank == 0)
    {
        cout << "m = " << m << endl;
    }
    return x;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // cout << "╰( ͡° ͜ʖ ͡° )つ──☆*" << endl;
    int n = 10;
    double tol = 1e-6;
    int maxit = 50;
    srand(time(nullptr));

    auto *b = new double[n];
    auto *x = new double[n];
    auto **A = new double *[n];

    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            A[i] = new double[n];
            b[i] = (-floatDist(gen));
            for (int j = 0; j < n; j++)
            {
                A[i][j] = (-floatDist(gen));
            }
        }
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    x = GMRES(A, b, n, maxit, tol);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    if (rank == 0)
    {
        std::cout << "Time for GMRES: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;
        for (int i = 0; i < n; i++)
        {
            cout << x[i] << ' ';
        }
        cout << endl;
    }

    // for (int i = 0; i < n; i++)
    // {
    //     delete[] A[i];
    // }
    // delete[] A; // Удаляем массивы
    // delete[] b;
    // delete[] x;

    MPI_Finalize();

    return 0;
}
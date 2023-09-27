#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <cblas.h>
#include <mpi.h>
#include <unistd.h>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> floatDist(-1, 0);

class Matvec
{
private:
    int row, col;
    MPI_Comm comm;
    int rank, size;
    std::vector<double> local_A;

    std::vector<int> shifts_rcv;
    std::vector<int> counts_rcv;

public:
    Matvec(const int n_, const int m_, MPI_Comm comm)
    {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        int local_n = n_ / size; // Число строк на каждом процессе

        if (rank == size - 1)
        {
            local_n += n_ % size;
        }

        local_A.resize(local_n * m_);
        shifts_rcv.resize(size);
        counts_rcv.resize(size);

        for (int i = 0; i < local_n * m_; i++)
        {
            local_A[i] = (-floatDist(gen));
        }

        row = local_n;
        col = m_;

        for (int i = 0; i < size; ++i)
        {
            counts_rcv[i] = (i != size - 1) ? n_ / size : n_ / size + n_ % size;
            shifts_rcv[i] = (n_ / size) * i;
        }
    }
    std::vector<double> matvec(std::vector<double> &x)
    {
        std::vector<double> local_result(row);

        cblas_dgemv(CblasRowMajor, CblasNoTrans, row, col, 1.0, local_A.data(), col, x.data(), 1, 1.0, local_result.data(), 1);

        std::vector<double> result(col);

        MPI_Gatherv(local_result.data(), row, MPI_DOUBLE, result.data(), counts_rcv.data(), shifts_rcv.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(result.data(), col, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        return result;
    }
};

std::vector<double> Solve_Upper_Triangular(std::vector<double> &R, std::vector<double> &b, int n)
{
    std::vector<double> x(n, 0);

    for (int i = n - 1; i >= 0; --i)
    {
        double sum = 0;
        for (int j = i + 1; j < n; ++j)
        {
            sum += R[i * n + j] * x[j];
        }
        x[i] = (b[i] - sum) / R[i * n + i];
    }
    return x;
}

double norm(std::vector<double> &y)
{
    double norm_local = std::sqrt(std::accumulate(y.begin(), y.end(), 0.0, [](double acc, double x)
                                                  { return acc + x * x; }));
    return norm_local;
}

std::vector<double> slice(std::vector<double> &y, int rows, int cols, int y_m)
{
    std::vector<double> res(rows * cols);

    for (int i = 0; i < rows; i++)
    {
        std::copy(y.begin() + i * y_m, y.begin() + i * y_m + cols, res.begin() + i * cols);
    }

    return res;
}

// find rotation to a-> (a^2 +b ^2)^0.5, b -> 0
void find_rot(double a, double b, double &c, double &s)
{
    // std::cout << a << " " << b << '\n';
    double mod = sqrt(a * a + b * b);
    if (mod == 0)
    {
        c = 1;
        s = 0;
    }
    else
    {
        s = -b / mod;
        c = a / mod;
    }
}

// apply all rotates to last col of matrix A.
void apply_rot_to_last_col(double *A, size_t m, size_t stride, std::vector<double> const &cs, std::vector<double> const &ss)
{
    size_t cur_pos = m - 1, next_pos = m + (stride - 1);
    for (int i = 0; i < cs.size(); ++i, cur_pos += stride, next_pos += stride)
    {
        double c = cs[i], s = ss[i];
        double a1 = c * A[cur_pos] - s * A[next_pos];
        double a2 = s * A[cur_pos] + c * A[next_pos];
        A[cur_pos] = a1, A[next_pos] = a2;
    }
}
void apply_rot_b(double *A, size_t m, size_t stride, std::vector<double> const &cs, std::vector<double> const &ss)
{
    size_t cur_pos = 0, next_pos = cur_pos + 1;
    for (int i = 0; i < cs.size(); ++i, cur_pos++, next_pos++)
    {
        double c = cs[i], s = ss[i];
        double a1 = c * A[cur_pos] - s * A[next_pos];
        double a2 = s * A[cur_pos] + c * A[next_pos];
        A[cur_pos] = a1, A[next_pos] = a2;
    }
}

std::vector<double> GMRES(std::vector<double> &b, Matvec &matvec, double atol = 1e-8, double rtol = 1e-6, int maxiter = 50, int proc_id = 0, int n_proc = 1)
{
    auto norm_b = norm(b); // norm_b = norm(b)
    std::vector<double> q(b.size());

    std::transform(b.begin(), b.end(), q.begin(), [norm_b](double value)
                   { return value / norm_b; }); // q = b / norm_b

    std::vector<double> Q = q; // Q = q[:, None]

    std::vector<double> h((maxiter + 1) * maxiter, 0); // h = np.zeros((maxiter + 1, maxiter))

    std::vector<double> y;

    std::vector<double> cos_rot, sin_rot;

    int n;
    double *r = new double[1];
    *r = norm_b;

    for (n = 0; n < maxiter; ++n)
    {
        MPI_Bcast(r, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(q.data(), q.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        auto v = matvec.matvec(q); // v = matvec(q)

        if (proc_id == 0)
        {
            for (int j = 0; j <= n; ++j)
            {
                std::vector<double> tmp(v.size());
                std::vector<double> Q_j(Q.begin() + q.size() * j, Q.begin() + q.size() * (j + 1));

                h[j * maxiter + n] = cblas_ddot(v.size(), Q_j.data(), 1, v.data(), 1); // h[j, n] = np.dot(Q[:, j], v)

                std::transform(Q_j.begin(), Q_j.end(), tmp.begin(), [&h, j, maxiter, n](double value)
                               { return value * h[j * maxiter + n]; }); // h[j, n] * Q[:, j]
                std::transform(v.begin(), v.end(), tmp.begin(), v.begin(),
                               [](double a, double b)
                               { return a - b; }); // v -= h[j, n] * Q[:, j]
            }
            h[(n + 1) * maxiter + n] = norm(v); // h[n+1, n] = norm(v)
            std::transform(v.begin(), v.end(), q.begin(), [&h, maxiter, n](double value)
                           { return value / h[(n + 1) * maxiter + n]; }); // q = v / h[n+1, n]
            Q.resize(Q.size() + q.size());
            std::copy(q.begin(), q.end(), Q.end() - q.size()); // Q = np.append(Q, q[:, None], axis=1)
            std::vector<double> e1(n + 2, 0);                  // e1 = np.zeros(n+2)
            e1[0] = norm_b;                                    // e1[0] = norm_b

            // QR algo
            apply_rot_to_last_col(h.data(), n + 1, maxiter, cos_rot, sin_rot); // apply all computed rotations to last h column

            double c, s;
            find_rot(h[n * maxiter + n], h[(n + 1) * maxiter + n], c, s); // find new rotation to turn h in triangular matrix

            double temp1 = h[n * maxiter + n] * c - h[(n + 1) * maxiter + n] * s;
            double temp2 = h[n * maxiter + n] * s + h[(n + 1) * maxiter + n] * c;
            h[n * maxiter + n] = temp1;
            h[(n + 1) * maxiter + n] = temp2;

            cos_rot.push_back(c), sin_rot.emplace_back(s);

            apply_rot_b(e1.data(), n + 2, 1, cos_rot, sin_rot); // apply all rotation to right side: Q.T @ b

            auto R = slice(h, n + 2, n + 1, maxiter); // h[:n+2, :n+1]
            int R_n = n + 2, R_m = n + 1;

            // // Lin solve start
            auto R_for_lin = slice(R, n + 1, n + 1, n + 1);
            *r = std::abs(e1.back()); // r = np.abs(Qb[-1])

            // std::cout << *r << " " << n << std::endl;

            std::vector<double> Qb_for_lin(n + 1);
            std::copy(e1.begin(), e1.begin() + n + 1, Qb_for_lin.begin());

            y = Solve_Upper_Triangular(R_for_lin, Qb_for_lin, n + 1);
            // // // Lin solve end
        }
        // std::cout << "here" << std::endl;
        // MPI_Bcast(r, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (*r < atol + norm_b * rtol)
        {
            // std::cout << *r << " " << n << std::endl;

            std::vector<double> Q_resized(Q.begin(), Q.begin() + (n + 1) * q.size());
            int Q_n = q.size(), Q_m = Q_resized.size() / q.size();

            std::vector<double> x(q.size(), 0);

            cblas_dgemv(CblasColMajor, CblasNoTrans, Q_n, Q_m, 1.0, Q_resized.data(), Q_n, y.data(), 1, 1.0,
                        x.data(), 1);
            MPI_Bcast(x.data(), x.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            return x;
        }
    }
    std::vector<double> x(q.size());
    if (proc_id == 0)
    {
        std::cout << "max iters reached: " << maxiter << std::endl;

        std::vector<double> Q_resized(Q.begin(), Q.begin() + (n + 1) * q.size());
        int Q_n = q.size(), Q_m = Q_resized.size() / q.size();

        cblas_dgemv(CblasColMajor, CblasNoTrans, Q_n, Q_m, 1.0, Q_resized.data(), Q_n, y.data(), 1, 1.0, x.data(), 1);
    }
    MPI_Bcast(x.data(), x.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return x;
}

int main(int argc, char **argv)
{
    int mpi_thread_prov;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_prov);
    int proc_id, n_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

    int n = 50;

    std::vector<double> b(n);

    if (proc_id == 0)
    {
        for (int i = 0; i < n; ++i)
        {
            b[i] = (-floatDist(gen));
        }
    }
    MPI_Bcast(b.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    Matvec matrix(n, n, MPI_COMM_WORLD);

    auto res = GMRES(b, matrix, 1e-8, 1e-6, n, proc_id, n_proc);

    auto tmp = matrix.matvec(res);

    if (proc_id == 0)
    {
        std::transform(tmp.begin(), tmp.end(), b.begin(), tmp.begin(), [](double a, double b)
                       { return a - b; });
        std::cout << "||Ax - b|| = " << norm(tmp) << std::endl;
    }

    MPI_Finalize();

    return 0;
}
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <lapacke.h>
#include <cblas.h>
#include "mpi.h"
#include <omp.h>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> floatDist(-1, 0);

class Matvec
{
private:
    int m, n;
    std::vector<double> A;

public:
    Matvec(const std::vector<double> mtx, const int n_, const int m_)
    {
        A = mtx;
        n = n_;
        m = m_;
    }

    std::vector<double> matvec(std::vector<double> &y)
    {
        std::vector<double> res(n);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, m, 1.0, A.data(), m, y.data(), 1, 1.0, res.data(), 1);
        return res;
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

double norm(std::vector<double> &y)
{
    return std::sqrt(std::accumulate(y.begin(), y.end(), 0.0, [](double acc, double x)
                                     { return acc + x * x; }));
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

std::vector<double> GMRES(std::vector<double> &b, Matvec &matvec, double atol = 1e-8, double rtol = 1e-6, int maxiter = 50)
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
    for (n = 0; n < maxiter; ++n)
    {
        auto v = matvec.matvec(q); // v = matvec(q)

        for (int j = 0; j <= n; ++j)
        {
            std::vector<double> tmp(v.size());
            std::vector<double> Q_j(Q.begin() + q.size() * j, Q.begin() + q.size() * (j + 1));

            h[j * maxiter + n] = cblas_ddot(v.size(), Q_j.data(), 1, v.data(), 1); // h[j, n] = np.dot(Q[:, j], v)

            std::transform(Q_j.begin(), Q_j.end(), tmp.begin(), [&h, j, maxiter, n](double value)
                           { return value * h[j * maxiter + n]; }); // h[j, n] * Q[:, j]
            std::transform(v.begin(), v.end(), tmp.begin(), v.begin(), [](double a, double b)
                           { return a - b; }); // v -= h[j, n] * Q[:, j]
        }

        h[(n + 1) * maxiter + n] = norm(v); // h[n+1, n] = norm(v)

        std::transform(v.begin(), v.end(), q.begin(), [&h, maxiter, n](double value)
                       { return value / h[(n + 1) * maxiter + n]; }); // q = v / h[n+1, n]

        Q.resize(Q.size() + q.size());
        std::copy(q.begin(), q.end(), Q.end() - q.size()); // Q = np.append(Q, q[:, None], axis=1)

        std::vector<double> e1(n + 2, 0); // e1 = np.zeros(n+2)
        e1[0] = norm_b;                   // e1[0] = norm_b

        // QR start
        apply_rot_to_last_col(h.data(), n + 1, maxiter, cos_rot, sin_rot); // apply all computed rotations to last h column

        double c, s;
        find_rot(h[n * maxiter + n], h[(n + 1) * maxiter + n], c, s); // find new rotation to turn h in triangular matrix

        double temp1 = h[n * maxiter + n] * c - h[(n + 1) * maxiter + n] * s;
        double temp2 = h[n * maxiter + n] * s + h[(n + 1) * maxiter + n] * c;
        h[n * maxiter + n] = temp1;
        h[(n + 1) * maxiter + n] = temp2;

        cos_rot.push_back(c), sin_rot.emplace_back(s);

        apply_rot_b(e1.data(), n + 2, 1, cos_rot, sin_rot); // apply all rotation to right side: Q.T @ b
        // QR end

        auto R = slice(h, n + 2, n + 1, maxiter); // h[:n+2, :n+1]

        int R_n = n + 2, R_m = n + 1;

        // // Lin solve start
        auto R_for_lin = slice(R, n + 1, n + 1, n + 1);
        double r = std::abs(e1.back()); // r = np.abs(Qb[-1])
        std::vector<double> Qb_for_lin(n + 1);
        std::copy(e1.begin(), e1.begin() + n + 1, Qb_for_lin.begin());

        y = Solve_Upper_Triangular(R_for_lin, Qb_for_lin, n + 1);
        // // // Lin solve end

        // std::cout << r << " " << n << std::endl;
        if (r < atol + norm_b * rtol)
        {

            std::vector<double> Q_resized(Q.begin(), Q.begin() + (n + 1) * q.size());
            int Q_n = q.size(), Q_m = Q_resized.size() / q.size();

            std::vector<double> x(q.size(), 0);

            cblas_dgemv(CblasColMajor, CblasNoTrans, Q_n, Q_m, 1.0, Q_resized.data(), Q_n, y.data(), 1, 1.0, x.data(), 1);

            return x;
        }
    }
    std::cout << "max iters reached: " << maxiter << std::endl;

    std::vector<double> Q_resized(Q.begin(), Q.begin() + (n + 1) * q.size());
    int Q_n = q.size(), Q_m = Q_resized.size() / q.size();

    std::vector<double> x(q.size());

    cblas_dgemv(CblasColMajor, CblasNoTrans, Q_n, Q_m, 1.0, Q_resized.data(), Q_n, y.data(), 1, 1.0, x.data(), 1);

    return x;
}

int main(int argc, char **argv)
{

    int n = 50, m = 50;
    std::vector<double> mtx(n * m);
    std::vector<double> b(m);

    for (int i = 0; i < n; ++i)
    {
        mtx[i * m + i] = (-floatDist(gen));
    }
    for (int i = 0; i < m; ++i)
    {
        b[i] = (-floatDist(gen));
    }

    Matvec matrix(mtx, n, m);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto res = GMRES(b, matrix, 1e-8, 1e-6, 1000);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // for (auto x : res)
    // {
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;

    auto tmp = matrix.matvec(res);
    std::transform(tmp.begin(), tmp.end(), b.begin(), tmp.begin(), [](double a, double b)
                   { return a - b; });
    std::cout << "||Ax - b|| = " << norm(tmp) << std::endl;

    return 0;
}
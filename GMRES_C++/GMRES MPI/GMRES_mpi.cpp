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
    int matrix_rows;
    MPI_Comm comm;
    int rank, size;

public:
    int row, col;
    std::vector<double> local_A;
    std::vector<int> shifts_rcv;
    std::vector<int> counts_rcv;
    std::vector<int> shifts_snd;
    std::vector<int> counts_snd;

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
        shifts_snd.resize(size);
        counts_rcv.resize(size);
        counts_snd.resize(size);

        for (int i = 0; i < local_n * m_; i++)
        {
            local_A[i] = (-floatDist(gen));
        }

        row = local_n;
        col = m_;
        matrix_rows = n_;

        for (int i = 0; i < size; ++i)
        {
            counts_rcv[i] = (i != size - 1) ? n_ / size : n_ / size + n_ % size;
            shifts_rcv[i] = (n_ / size) * i;
            shifts_snd[i] = shifts_rcv[i] * col;
            counts_snd[i] = counts_rcv[i] * col;
        }
    }
    void append(std::vector<double> &x)
    {
        col++;
        local_A.resize(row * col);

        std::copy(x.begin(), x.end(), local_A.end() - x.size());
    }

    std::vector<double> matvec(std::vector<double> &x, CBLAS_ORDER order = CblasRowMajor)
    {
        std::vector<double> local_result(row);

        int lda = (order == CblasRowMajor) ? col : row;

        cblas_dgemv(order, CblasNoTrans, row, col, 1.0, local_A.data(), lda, x.data(), 1, 1.0, local_result.data(), 1);

        std::vector<double> result(matrix_rows);

        MPI_Gatherv(local_result.data(), row, MPI_DOUBLE, result.data(), counts_rcv.data(), shifts_rcv.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(result.data(), result.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        return result;
    }

    std::vector<double> matvec_2(std::vector<double> &x_chunk, CBLAS_ORDER order = CblasRowMajor) {
        std::vector<double> local_result(row);

        int lda = (order == CblasRowMajor) ? col : row;
        std::vector<int> x_sizes(size);
        int tmp = x_chunk.size();
        MPI_Allgather(&tmp, 1, MPI_INT, x_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

        auto matvec_chunk = x_chunk;

        //cblas_dgemv(order, CblasNoTrans, row, col, 1.0, local_A.data(), lda, x.data(), 1, 1.0, local_result.data(), 1);

        for (int shift = 1; shift < size; ++shift){
            int send_proc_id = (rank + shift) % size, recv_proc_id = (rank - shift + size) % size;
            std:: vector<double> next_chunk(x_sizes[recv_proc_id]);
            MPI_Request send, recv;
            MPI_Isend(x_chunk.data(), x_chunk.size(), MPI_DOUBLE, send_proc_id, shift, MPI_COMM_WORLD, &send);
            MPI_Irecv(next_chunk.data(), next_chunk.size(), MPI_DOUBLE, recv_proc_id, shift, MPI_COMM_WORLD, &recv);
            int block_start_col = (rank + shift - 1) % size;
            std::vector<double> A_chunk(row * x_chunk.size());
            for (int i = 0; i < row; ++i){
                std::copy(local_A.begin() + i * col + block_start_col, local_A.begin() + i * col + block_start_col + x_chunk.size(), A_chunk.begin() + i * x_chunk.size());
            }
            cblas_dgemv(order, CblasNoTrans, row, x_chunk.size(), 1.0, A_chunk.data(), lda, matvec_chunk.data(), 1, 1.0, local_result.data(), 1);
            //cblas_dgemv (order, CblasNoTrans, n, n, 1.0, &A_chunk(0, block_start_col * n), k, matvec_chunk.data(), 1, 1, y_chunk.data(), 1);
            MPI_Status status;
            MPI_Wait(&send, &status);
            MPI_Wait(&recv, &status);
            matvec_chunk = next_chunk;
        }

        //std::vector<double> result(matrix_rows);

        //MPI_Gatherv(local_result.data(), row, MPI_DOUBLE, result.data(), counts_rcv.data(), shifts_rcv.data(),
        //            MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //MPI_Bcast(result.data(), result.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        return local_result;
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
double squares(std::vector<double> &y)
{
    double norm_local = std::accumulate(y.begin(), y.end(), 0.0, [](double acc, double x)
                                        { return acc + x * x; });
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
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Matvec Q_(matvec.col, 1, MPI_COMM_WORLD);

    double norm_b = squares(b);
    MPI_Allreduce(MPI_IN_PLACE, &norm_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    norm_b = std::sqrt(norm_b); // norm_b = norm(b)

    std::vector<double> q(b.size());

    std::transform(b.begin(), b.end(), q.begin(), [norm_b](double value)
                   { return value / norm_b; }); // q = b / norm_b
    //MPI_Scatterv(q.data(), Q_.counts_snd.data(), Q_.shifts_snd.data(), MPI_DOUBLE, Q_.local_A.data(), q.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD); // разбили Q по процессам.
    std::copy(q.begin(), q.end(), Q_.local_A.begin());
    std::vector<double> h((maxiter + 1) * maxiter, 0); // h = np.zeros((maxiter + 1, maxiter))

    std::vector<double> y;
    std::vector<double> x;
    std::vector<double> cos_rot, sin_rot;

    int n;
    double r = norm_b;

    for (n = 0; n < maxiter; ++n)
    {
        MPI_Bcast(&r, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //MPI_Bcast(q.data(), q.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        auto v = matvec.matvec_2(q); // v = matvec(q)

        //std::vector<double> v_slice(v.begin() + Q_.shifts_snd[proc_id], v.begin() + Q_.shifts_snd[proc_id] + Q_.counts_snd[proc_id]);

        for (int j = 0; j <= n; ++j)
        {
            std::vector<double> tmp(v.size(), 0);
            std::vector<double> Q_j(Q_.local_A.begin() + Q_.row * j, Q_.local_A.begin() + Q_.row * (j + 1));

            double local, global = 0;

            local = cblas_ddot(Q_j.size(), Q_j.data(), 1, v.data(), 1);

            MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Bcast(&global, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            h[j * maxiter + n] = global; // h[j, n] = np.dot(Q[:, j], v)

            std::transform(Q_j.begin(), Q_j.end(), tmp.begin(), [&global](double value)
                           { return value * global; }); // h[j, n] * Q[:, j]

            std::transform(v.begin(), v.end(), tmp.begin(), v.begin(),
                           [](double a, double b)
                           { return a - b; }); // v -= h[j, n] * Q[:, j]

//            MPI_Gatherv(v.data(), v.size(), MPI_DOUBLE, v.data(), Q_.counts_rcv.data(), Q_.shifts_rcv.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            // MPI_Bcast(v.data(), v.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD)
        }

        // PARALLEL NORM OF v start
        double h_n_1_n = squares(v);
        MPI_Allreduce(MPI_IN_PLACE, &h_n_1_n, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        h_n_1_n = std::sqrt(h_n_1_n);
        //  MPI_Bcast(&h_n_1_n, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // PARALLEL NORM OF v end

        h[(n + 1) * maxiter + n] = h_n_1_n;

        std::vector<double> q_slice(v.size());

        std::transform(v.begin(), v.end(), q_slice.begin(), [&h_n_1_n](double value)
                       { return value / h_n_1_n; });

        Q_.append(q_slice);

//        MPI_Gatherv(q_slice.data(), q_slice.size(), MPI_DOUBLE, q.data(), Q_.counts_rcv.data(), Q_.shifts_rcv.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (proc_id == 0)
        {
            std::vector<double> e1(n + 2, 0); // e1 = np.zeros(n+2)
            e1[0] = norm_b;                   // e1[0] = norm_b

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
            // QR end
            auto R = slice(h, n + 2, n + 1, maxiter); // h[:n+2, :n+1]

            int R_n = n + 2, R_m = n + 1;

            // // Lin solve start
            auto R_for_lin = slice(R, n + 1, n + 1, n + 1);
            r = std::abs(e1.back()); // r = np.abs(Qb[-1])

            std::vector<double> Qb_for_lin(n + 1);
            std::copy(e1.begin(), e1.begin() + n + 1, Qb_for_lin.begin());

            y = Solve_Upper_Triangular(R_for_lin, Qb_for_lin, n + 1);
            // // // Lin solve end
        }

        if (proc_id)
        {
            y.resize(n + 1);
        }

        MPI_Bcast(y.data(), y.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&r, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (r < atol + norm_b * rtol)
        {
            if (proc_id == 0)
            {
                std::cout << "r: " << r << " n: " << n << std::endl;
            }

            Q_.local_A = std::vector(Q_.local_A.begin(), Q_.local_A.begin() + (n + 1) * Q_.row);
            Q_.col = n + 1;

            x = Q_.matvec(y, CblasColMajor);

            return x;
        }
    }

    Q_.local_A = std::vector(Q_.local_A.begin(), Q_.local_A.begin() + (n + 1) * Q_.row);
    Q_.col = n + 1;

    x = Q_.matvec(y, CblasColMajor);

    return x;
}

int main(int argc, char **argv)
{
    int mpi_thread_prov;
    const double t = 1e-8;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_prov);
    int proc_id, n_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

    int n = 20;

    // std::vector<double> b(n);
    int b_size = n / n_proc; // Число строк на каждом процессе

    if (proc_id == n_proc - 1)
    {
        b_size += n % n_proc;
    }

    std::vector<double> b(b_size);
    for (auto &x : b)
    {
        x = (-floatDist(gen));
    }

    Matvec matrix(n, n, MPI_COMM_WORLD);

    auto res = GMRES(b, matrix, t, 1e-6, n, proc_id, n_proc);

    auto tmp = matrix.matvec_2(res);


    std::transform(tmp.begin(), tmp.end(), b.begin(), tmp.begin(), [](double a, double b)
                   { return a - b; });

    double norm_res = squares(tmp);
    MPI_Allreduce(MPI_IN_PLACE, &norm_res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    norm_res = std::sqrt(norm_res);

    if (proc_id == 0){
        std::cout << "||Ax - b|| = " << norm_res << std::endl;
    }

    MPI_Finalize();

    return 0;
}
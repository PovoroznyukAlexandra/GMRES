#include <iostream>
#include <vector>
#include <cmath>


using namespace std;

// Функция для умножения матрицы на вектор
double* MatVec(double** A, int n, const double* x, int m) {
    auto* y = new double[n];
    for (int i = 0; i < n; i++) {
        y[i] = 0;
        for (int j = 0; j < m; j++) {
            y[i] += A[i][j] * x[j];
        }
    }
    return y;
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
        v = MatVec(A, n, v ,n);
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
            x = MatVec(V_1, n, c, m-1);
            /*
            for (int i = 0; i < n; i++){
                cout << x[i] << ' ';
            }
            cout << endl;
             */
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
        x = MatVec(V_2, n, c, m);
    }
    cout << "m = " << m << endl;
    //return x, residual
    for (int i = 0; i < n; i++){
        cout << x[i] << ' ';
    }
    return x;
}

int main() {
    // Создание матрицы A
    double tol = 1e-6;
    int maxit = 100;
    //int n = 4;
    int n;
    cout << "Enter N: (n = 4)" << endl;
    cin >> n;
    auto **A = new double* [n];  //H = QR
    for (int i = 0; i < n; i++){
        A[i] = new double [n];
        for (int j = 0; j < n; j++){
            A[i][j] = 0;
        }
    }
    A[0][0] = 1;
    A[1][1] = -1;
    A[2][2] = 2;
    A[3][3] = -2;
    cout << "Matrix A: " << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << A[i][j] << ' ';
        }
        cout << endl;
    }
    cout << endl;

    // Создание вектора b
    cout << "Vector b: " << endl;
    auto *b = new double[n];
    for (int i = 0; i < n; i++){
        b[i] = 1;
        cout << b[i] << ' ';
    }
    cout << endl;
    cout << endl;

    auto *x = new double[n];
    for (int i = 0; i < n; i++){
        x[i] = 0;
    }
    x = GMRES(A, b, n, maxit, tol);

    cout << "Solution: " << endl;
    for (int i = 0; i < n; i++){
        cout << x[i] << ' ';
    }
    cout << endl << endl;


    auto *err = new double[n];
    err = MatVec(A, n, x, n);
    for (int i = 0; i < n; i++){
        err[i] -= b[i];
    }
    cout << "Error: " << Norm(err, n) << endl;

    for (int i = 0; i < n; i++){
        delete[] A[i]; // Удаляем каждый элемент
    }
    delete[] A;
    delete[] b;
    delete[] x;
    //delete[] err;
    return 0;

}
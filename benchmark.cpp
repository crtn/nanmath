#include <iostream>
#include <iomanip>
#include <chrono>
#include <stack>
#include <armadillo>
#include "nanmath.h"

class Clock {
public:
    void tic() {
        ts.push(std::chrono::high_resolution_clock::now());
    }

    long toc() {
        if(ts.empty()) {
            throw "unmatched tic/toc";
        }
        auto t0 = ts.top();
        ts.pop();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    }
private:
    std::stack<std::chrono::high_resolution_clock::time_point> ts;
};

int main(int argc, char *argv[]) {
    if(argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_matrix>" << std::endl;
        return -1;
    }

    Clock clk;
    arma::Mat<float> mf;
    arma::Mat<double> md;
    mf.load(argv[1]);
    md.load(argv[1]);

    unsigned n_rows = mf.n_rows, n_cols = mf.n_cols;
    std::cout << "processing " << n_rows << " x " << n_cols << " matrix" << std::endl;
    double sum = 0.0;
    unsigned count = 0;

    clk.tic();
    {
        sum = 0.0;
        count = 0;
        float corr;
        unsigned cnt;
        for(unsigned i = 0; i < n_cols; ++i) {
            for(unsigned j = i + 1; j < n_cols; ++j) {
                nan_corr_float(mf.colptr(i), mf.colptr(j), n_rows, &corr, &cnt);
                sum += corr;
                count += cnt;
            }
        }
    }
    std::cout << std::setw(32) << std::right << "nan_corr_float: "
              << std::setw(12) << std::left << std::to_string(clk.toc()) + " ms"
              << std::setw(12) << std::right << "sum: "
              << std::setw(12) << std::left << sum
              << std::setw(12) << std::right << "count: "
              << std::setw(12) << std::left << count << std::endl;

    clk.tic();
    {
        sum = 0.0;
        count = 0;
        float corr;
        unsigned cnt;
        for(unsigned i = 0; i < n_cols; ++i) {
            for(unsigned j = i + 1; j < n_cols; ++j) {
                nan_corr_float_avx(mf.colptr(i), mf.colptr(j), n_rows, &corr, &cnt);
                sum += corr;
                count += cnt;
            }
        }
    }
    std::cout << std::setw(32) << std::right << "nan_corr_float_avx: "
              << std::setw(12) << std::left << std::to_string(clk.toc()) + " ms"
              << std::setw(12) << std::right << "sum: "
              << std::setw(12) << std::left << sum
              << std::setw(12) << std::right << "count: "
              << std::setw(12) << std::left << count << std::endl;

    clk.tic();
    {
        sum = 0.0;
        count = 0;
        double corr;
        unsigned cnt;
        for(unsigned i = 0; i < n_cols; ++i) {
            for(unsigned j = i + 1; j < n_cols; ++j) {
                nan_corr_double(md.colptr(i), md.colptr(j), n_rows, &corr, &cnt);
                sum += corr;
                count += cnt;
            }
        }
    }
    std::cout << std::setw(32) << std::right << "nan_corr_double: "
              << std::setw(12) << std::left << std::to_string(clk.toc()) + " ms"
              << std::setw(12) << std::right << "sum: "
              << std::setw(12) << std::left << sum
              << std::setw(12) << std::right << "count: "
              << std::setw(12) << std::left << count << std::endl;

    clk.tic();
    {
        sum = 0.0;
        count = 0;
        double corr;
        unsigned cnt;
        for(unsigned i = 0; i < n_cols; ++i) {
            for(unsigned j = i + 1; j < n_cols; ++j) {
                nan_corr_double_avx(md.colptr(i), md.colptr(j), n_rows, &corr, &cnt);
                sum += corr;
                count += cnt;
            }
        }
    }
    std::cout << std::setw(32) << std::right << "nan_corr_double_avx: "
              << std::setw(12) << std::left << std::to_string(clk.toc()) + " ms"
              << std::setw(12) << std::right << "sum: "
              << std::setw(12) << std::left << sum
              << std::setw(12) << std::right << "count: "
              << std::setw(12) << std::left << count << std::endl;

    return 0;
}

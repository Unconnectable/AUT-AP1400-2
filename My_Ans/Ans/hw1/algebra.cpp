#include "algebra.h"
#include <vector>
#include <random>
//#include <format>
#include <fmt/core.h>
#include <iostream>
#include <stdexcept>
#include <optional>

using std::size_t;
using std::vector;

namespace algebra {

    //create_matrix<int>(2, 2, MatrixType::Random, 5, 4)
    template<typename T>
    MATRIX<T> create_matrix(
        std::size_t rows, 
        std::size_t columns,
        std::optional<MatrixType> type,
        
        //enum class MatrixType { Zeros, Ones, Identity, Random };

        //create_matrix<int>(2, 2, MatrixType::Random, 5, 4)
        std::optional<T> lowerBound,
        std::optional<T> upperBound) {

        MATRIX<T> matrix(rows, std::vector<T>(columns));
        if (type == MatrixType::Zeros){
            if((int)rows == 0 || (int)columns ==0)
                throw std::invalid_argument("Random Type ,lowerBound or upperBound have some error");
            for(auto &row : matrix)
                for(auto &elem : row)
                    elem = 0;
        }
        else if(type == MatrixType::Ones){
            for(auto &row : matrix)
                for(auto &elem : row)
                    elem = 1;
        }
        else if(type == MatrixType::Identity){
            if(rows != columns)
                throw std::invalid_argument("rows not equal to columns");
            for(size_t i = 0; i < rows ;++i){
                matrix[i][i] = 1;
            }
        }
        else {
            if(lowerBound == std::nullopt || upperBound == std::nullopt || lowerBound.value() > upperBound.value())
                throw std::invalid_argument("Random Type ,lowerBound or upperBound have some error");

            std::random_device rd;  
            std::mt19937 gen(rd()); 
            std::uniform_real_distribution<double> dis(lowerBound.value(),upperBound.value());
            for(auto &row : matrix)
                for(auto &elem : row)
                    elem = dis(gen);
        
        }
        return matrix;
    }
    
    template<typename T>
    void display(const MATRIX<T>& matrix){
        for(auto & row : matrix){
            for(auto &elem : row){
                std::cout << "|" << fmt::format("{:^7}",elem);
            }
            std::cout << "|" << std::endl;
        }
    }

    template<typename T>
    MATRIX<T> sum_sub(
        const MATRIX<T>& matrixA, 
        const MATRIX<T>& matrixB,
        std::optional<std::string> operation){
            if (matrixA.empty() || matrixB.empty() || matrixA[0].empty() || matrixB[0].empty()) {
                return {};  // 返回一个空矩阵
            }
            
            std::size_t rows1 = matrixA.size();
            std::size_t cols1 = matrixA[0].size();
            std::size_t rows2 = matrixB.size();
            std::size_t cols2 = matrixB[0].size();
            if(rows1 != rows2 || cols1 != cols2)
                throw std::invalid_argument("The Size of MatrixA and MatrixB are not the same");
            MATRIX<T>   matrixC(matrixA.size(), std::vector<T>  (matrixA[0].size(), 0));

            if(operation.has_value() && operation.value() == "sub"){
                for(size_t i = 0 ; i < matrixA.size() ; i++){
                    for(size_t j = 0 ; j < matrixA[0].size() ; j++){
                        matrixC[i][j] = matrixA[i][j] - matrixB[i][j];
                    }
                }
            }
            else {
                for(size_t i = 0 ; i < matrixA.size() ; i++){
                    for(size_t j = 0 ; j < matrixA[0].size() ; j++){
                        matrixC[i][j] = matrixA[i][j] + matrixB[i][j];
                        
                    }
                }
            }

            return matrixC;
        }
        template<typename T>
        MATRIX<T> multiply(
            const MATRIX<T>& matrix, 
            const T scalar){
                MATRIX<T> matrix_ = matrix;
                for(auto& rows : matrix_){
                    for(auto& elem: rows){
                        elem = elem * scalar;
                    }
                }
                return matrix_;
            }

        template<typename T>
        MATRIX<T> multiply(
            const MATRIX<T>& matrixA, 
            const MATRIX<T>& matrixB){
                int r  = matrixA.size();      // A 的行数
                int s  = matrixA[0].size();   // A 的列数 (B 的行数)
                int s_ = matrixB.size();
                int t  = matrixB[0].size();   // B 的列数
                if(s != s_)
                    throw std::invalid_argument("The cols of MatrixA not match the rows of MatrixB");

                MATRIX<T> matrixC(r, std::vector<T>(t, 0));
                for (int i = 0; i < r; ++i) {
                    for (int j = 0; j < t; ++j) {
                        for (int k = 0; k < s; ++k) {
                            matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
                        }
                    }
                }

                return matrixC;
            }

    template<typename T>
    MATRIX<T> hadamard_product(
        const MATRIX<T>& matrixA, 
        const MATRIX<T>& matrixB){
            int r   = matrixA.size();      // A 的行数
            int s   = matrixA[0].size();   // A 的列数 (B 的行数)
            int k   = matrixB.size();
            int t   = matrixB[0].size();   // B 的列数
            if(r!=k || s!=t)
                throw std::invalid_argument("The cols of MatrixA not match the rows of MatrixB");
            MATRIX<T> matrixC(r,std::vector<T>(s,0));
            for (int i = 0; i < r; ++i) 
                for (int j = 0; j < t; ++j) 
                    matrixC[i][j] = matrixC[i][j] * matrixC[i][j];

            return matrixC;
        }

    template<typename T>
    MATRIX<T> transpose(
        const MATRIX<T>& matrix){
            int r  = matrix.size();      // A 的行数
            int s  = matrix[0].size();   // A 的列数 (B 的行数)
            MATRIX<T> matrix_(s,std::vector<T>(r,0));
            for (int i = 0; i < s; ++i) 
                for (int j = 0; j < r; ++j) 
                    matrix_[i][j] = matrix[j][i];
            return matrix_;
        }
    template<typename T>
    T trace(
        const MATRIX<T>& matrix){
            T trace_ = 0;
            int rows  = matrix.size();      // A 的行数
            int colos  = matrix.size();   // A 的列数 (B 的行数)
            if(rows != colos)
                throw std::invalid_argument("The matrix is not Identity");
            for(int i = 0; i < rows ; ++i){
                trace_ += matrix[i][i];
            }
        return trace_;
        }

    template<typename T>
    double determinant(const MATRIX<T>& matrix) {
        int rows = matrix.size();       // A 的行数
        int cols = matrix[0].size();    // A 的列数
        if (rows != cols)
            throw std::invalid_argument("The matrix is not square.");

        // 2x2 矩阵的行列式计算
        if (rows == 2)
            return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1];

        double det = 0.0;
        for (int j = 0; j < cols; ++j) {
            // 创建子矩阵 A_ij，通过去掉第0行和第j列
            MATRIX<T> subMatrix;
            for (int i = 1; i < rows; ++i) {
                std::vector<T> row;  // 将当前行初始化为空的 vector
                for (int k = 0; k < cols; ++k) {
                    if (k != j) {   // 去掉第 j 列
                        row.push_back(matrix[i][k]);
                    }
                }
                subMatrix.push_back(row);
            }

            // 递归计算子矩阵的行列式
            det += ((j % 2 == 0 ? 1 : -1) * matrix[0][j] * determinant(subMatrix));
        }
        return det;
    }

    template<typename T>
    MATRIX<double> inverse(const MATRIX<T>& matrix) {
        int n = matrix.size();

        // Ensure the matrix is square
        if (n == 0 || (int)matrix[0].size() != n) {
            throw std::invalid_argument("Matrix must be square.");
        }

        // Create augmented matrix [matrix | I]
        MATRIX<double> A(n, std::vector<double>(n));
        MATRIX<double> I(n, std::vector<double>(n, 0));

        // Copy input matrix into A and identity matrix into I
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i][j] = static_cast<double>(matrix[i][j]);
                if (i == j) {
                    I[i][j] = 1.0;  // Identity matrix on the diagonal
                }
            }
        }

        // Gaussian-Jordan elimination
        for (int i = 0; i < n; ++i) {
            // Find the row with the largest pivot
            int max_row = i;
            for (int j = i + 1; j < n; ++j) {
                if (abs(A[j][i]) > abs(A[max_row][i])) {
                    max_row = j;
                }
            }

            // If the pivot is zero, the matrix is singular and cannot be inverted
            if (A[max_row][i] == 0) {
                throw std::invalid_argument("Matrix is singular and cannot be inverted.");
            }

            // Swap rows in both A and I
            swap(A[i], A[max_row]);
            swap(I[i], I[max_row]);

            // Normalize the pivot row
            double pivot = A[i][i];
            for (int j = 0; j < n; ++j) {
                A[i][j] /= pivot;
                I[i][j] /= pivot;
            }

            // Eliminate other rows
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    double factor = A[j][i];
                    for (int k = 0; k < n; ++k) {
                        A[j][k] -= factor * A[i][k];
                        I[j][k] -= factor * I[i][k];
                    }
                }
            }
        }
        // Return the inverse matrix
        return I;
    }

        // create_matrix 的显式实例化
    template MATRIX<int> create_matrix<int>(std::size_t, std::size_t, std::optional<MatrixType>, std::optional<int>, std::optional<int>);
    template MATRIX<double> create_matrix<double>(std::size_t, std::size_t, std::optional<MatrixType>, std::optional<double>, std::optional<double>);
    template MATRIX<float> create_matrix<float>(std::size_t, std::size_t, std::optional<MatrixType>, std::optional<float>, std::optional<float>);

    // display 的显式实例化
    template void display<int>(const MATRIX<int>&);
    template void display<double>(const MATRIX<double>&);
    template void display<float>(const MATRIX<float>&);

    // sum_sub 的显式实例化
    template MATRIX<int> sum_sub<int>(const MATRIX<int>&, const MATRIX<int>&, std::optional<std::string>);
    template MATRIX<double> sum_sub<double>(const MATRIX<double>&, const MATRIX<double>&, std::optional<std::string>);
    template MATRIX<float> sum_sub<float>(const MATRIX<float>&, const MATRIX<float>&, std::optional<std::string>);

    // multiply (scalar) 的显式实例化
    template MATRIX<int> multiply<int>(const MATRIX<int>&, const int);
    template MATRIX<double> multiply<double>(const MATRIX<double>&, const double);
    template MATRIX<float> multiply<float>(const MATRIX<float>&, const float);

    // multiply (matrix-matrix) 的显式实例化
    template MATRIX<int> multiply<int>(const MATRIX<int>&, const MATRIX<int>&);
    template MATRIX<double> multiply<double>(const MATRIX<double>&, const MATRIX<double>&);
    template MATRIX<float> multiply<float>(const MATRIX<float>&, const MATRIX<float>&);

    // hadamard_product 的显式实例化
    template MATRIX<int> hadamard_product<int>(const MATRIX<int>&, const MATRIX<int>&);
    template MATRIX<double> hadamard_product<double>(const MATRIX<double>&, const MATRIX<double>&);
    template MATRIX<float> hadamard_product<float>(const MATRIX<float>&, const MATRIX<float>&);

    // transpose 的显式实例化
    template MATRIX<int> transpose<int>(const MATRIX<int>&);
    template MATRIX<double> transpose<double>(const MATRIX<double>&);
    template MATRIX<float> transpose<float>(const MATRIX<float>&);

    // trace 的显式实例化
    template int trace<int>(const MATRIX<int>&);
    template double trace<double>(const MATRIX<double>&);
    template float trace<float>(const MATRIX<float>&);

    // determinant 的显式实例化
    template double determinant<int>(const MATRIX<int>&);
    template double determinant<double>(const MATRIX<double>&);
    template double determinant<float>(const MATRIX<float>&);

    // inverse 的显式实例化
    template MATRIX<double> inverse<int>(const MATRIX<int>&);
    template MATRIX<double> inverse<double>(const MATRIX<double>&);
    template MATRIX<double> inverse<float>(const MATRIX<float>&);


}
#pragma once

#include <armadillo>

// Counts the number of gene groups (defined by rows in `ranges`) that contain
// at least one non-zero coefficient in `arr`.
// `ranges` is an n_genes × 2 matrix of 1-based [start, end] column indices.
template<typename T>
int countNonZeroGenes(const arma::Col<T>& arr, const arma::mat& ranges, const arma::Row<T>& field) {
    auto detectNonZeroInRange = [&arr](int start, int end, const arma::Row<T>& field) -> int {
        for (int i = start; i <= end; ++i) {
            if (arr(field[i]) != 0) {
                return 1;
            }
        }
        return 0;
    };
    int count = 0;

    for (arma::uword i = 0; i < ranges.n_rows; ++i) {
        int start = static_cast<int>(ranges(i, 0))-1;
        int end = static_cast<int>(ranges(i, 1))-1;
        count = count + detectNonZeroInRange(start, end, field);
    }

    //std::cout << "Number of non-zero genes: " << count << std::endl;
    return count;
}

template<typename T>
int countNonZeroGenes(const arma::Col<T>& arr, const arma::mat& ranges) {
    auto detectNonZeroInRange = [&arr](int start, int end) -> int {
        for (int i = start; i <= end; ++i) {
            if (arr(i) != 0) {
                return 1;
            }
        }
        return 0;
    };
    int count = 0;

    for (arma::uword i = 0; i < ranges.n_rows; ++i) {
        int start = static_cast<int>(ranges(i, 0))-1;
        int end = static_cast<int>(ranges(i, 1))-1;
        count = count + detectNonZeroInRange(start, end);
    }

    //std::cout << "Number of non-zero genes: " << count << std::endl;
    return count;
}

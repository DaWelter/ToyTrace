#pragma once

namespace Eigen
{

inline static constexpr struct eigen_zero_initialize_t 
{

} zero;

inline static constexpr struct eigen_ones_initialize_t
{
} ones;

}

#define EIGEN_MATRIX_PLUGIN "eigen_matrix_plugin.hxx"
#define EIGEN_ARRAY_PLUGIN "eigen_array_plugin.hxx"

#include <Eigen/Core>
#pragma once

#include "cubature.h"
#include "vec3f.hxx"

namespace cubature_wrapper
{

template<class Func>
inline int f1d(unsigned ndim, const double *x, void *fdata_,
      unsigned fdim, double *fval)
{
  auto* callable = static_cast<Func*>(fdata_);
  assert(ndim == 2);
  assert(fdim == 1);
  Eigen::Map<const Eigen::Vector2d> xm{x};
  *fval = (*callable)(xm);
  return 0;
}

template<class Func>
inline int fnd(unsigned ndim, const double *x, void *fdata_,
        unsigned fdim, double *fval)
{
  auto* callable = static_cast<Func*>(fdata_);
  assert(ndim == 2);
  Eigen::Map<const Eigen::Vector2d> xm{x};
  const auto value = ((*callable)(xm)).eval();
  std::copy(value.data(), value.data()+fdim, fval);
  return 0;
}

} // namespace cubature_wrapper


namespace 
{
  

template<class Func>
inline auto Integral2D(Func func, Double2 start, Double2 end, double absError, double relError, int max_eval = 0, decltype(func(Double2{})) *error_estimate = nullptr)
{
  using R = decltype(func(Double2{}));
  int status = -1;
  R result, error;
  if constexpr (std::is_floating_point<R>::value)
  {
    status = hcubature(1, cubature_wrapper::f1d<Func>, &func, 2, start.data(), end.data(), max_eval, absError, relError, ERROR_L2, &result, &error);
  }
  else
  {
    status = hcubature(result.size(), cubature_wrapper::fnd<Func>, &func, 2, start.data(), end.data(), max_eval, absError, relError, ERROR_L2, result.data(), error.data());
  }
  if (status != 0)
    throw std::runtime_error("Cubature failed!");
  if (error_estimate)
    *error_estimate = error;
  return result;
}


} // namespace

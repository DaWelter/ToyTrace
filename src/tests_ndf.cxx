#include "cubature_wrapper.hxx"
#include "cubemap.hxx"
#include "tests_stats.hxx"

#include "normaldistributionfunction.hxx"
#include "util.hxx"


template<class Func>
auto IntegralOverCubemap(Func func, const CubeMap &cubemap, double absError, double relError, int max_eval = 0)
{
  using R = decltype(func(Double2{}));
  std::vector<std::pair<R,R>> result;  result.reserve(cubemap.TotalNumBins());
  for (int idx=0; idx<cubemap.TotalNumBins(); ++idx)
  {
    auto [side, i, j] = cubemap.IndexToCell(idx);
    // Integrating this gives me the probability of a sample falling
    // into the current bin. The integration is carried out on a square in
    // R^2, hence the scale factor J from the Jacobian of the variable transform.
    auto probabilityDensityTimesJ = [side = side,&cubemap,&func](const Double2 x) -> double
    {
      Double3 omega = cubemap.UVToOmega(side, x);
      R val = func(omega)*cubemap.UVtoJ(x);
      return val;
    };
    auto [start, end] = cubemap.CellToUVBounds(i, j);
    R err = NaN;
    const R prob = Integral2D(probabilityDensityTimesJ, start, end, absError, relError, max_eval, &err);
    result.emplace_back(err, prob);
  }
  return result;
}


void TestBeckman()
{
  using NDF = BeckmanDistribution;
  CubeMap cubemap(8);
  const double alpha = 0.05;
  
  NDF ndf{alpha};

  auto density_func = [&](const Double3 &w)
  {
    // Argument is w.dot.n, where n is aligned with the z-axis.
    return ndf.EvalByHalfVector(w[2]);
  };

  constexpr int MAX_NUM_FUNC_EVALS = 100000;  
  std::vector<std::pair<double,double>> probs = IntegralOverCubemap(density_func, cubemap, 1.e-3, 1.e-2, MAX_NUM_FUNC_EVALS);
}

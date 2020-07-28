#pragma once

#include "vec3f.hxx"
#include "util.hxx"
#include "span.hxx"

#include "pcg32/pcg32.h"

#include <cmath>
#include <array>
#include <algorithm>
#include <numeric>

namespace SampleTrafo
{
Double3 ToUniformDisc(Double2 r); // In the x-y plane
Double3 ToUniformSphere(Double2 r);
Double3 ToUniformHemisphere(Double2 r);
Double3 ToUniformSphereSection(double cos_opening_angle, Double2 r);
double UniformSphereSectionPdf(double cos_opening_angle);
Double3 ToUniformSphereSection(Double2 r, double phi0, double z0, double phi1, double z1);
Double3 ToCosHemisphere(Double2 r);
Double3 ToPhongHemisphere(Double2 r, double alpha);
Double3 ToTriangleBarycentricCoords(Double2 r);
Double3 ToUniformSphere3d(const Double3 &r);
Float2 ToNormal2d(const Float2 &r);
}


class RandGen
{
  pcg32 generator;
public:
  RandGen();
  void Seed(std::uint64_t seed);

  void Uniform01(double *dest, int count);
  int UniformInt(int a, int b_inclusive);

  inline double Uniform01()
  {
    double r;
    Uniform01(&r, 1);
    return r;
  }

  inline Double2 UniformUnitSquare()
  {
    Double2 r;
    Uniform01(r.data(), 2);
    return r;
  }

  static constexpr std::uint64_t default_seed = PCG32_DEFAULT_STATE;

  pcg32& GetGenerator()
  {
    return generator;
  }
};


double SobolSequence(int point, int dim);


class SampleSequence
{
public:
  virtual ~SampleSequence() {}

  virtual void SetPixelIndex(Int2 pixel_coord) {}
  // Within the sequence. Normally incremented once per pass, i.e. for each sample per pixel.
  virtual void SetPointNum(int i) {}
  // Normally the dimension
  virtual void SetSubsequenceId(uint32_t id) {}

  virtual double Uniform01() = 0;
  virtual Double2 UniformUnitSquare() = 0;
};


class Sampler
{
  RandGen randgen;
  std::unique_ptr<SampleSequence> sequence;
public:
  explicit Sampler(bool use_qmc_sequence=false);

  RandGen& GetRandGen() { return randgen; }

  void Seed(std::uint64_t seed) { randgen.Seed(seed); }

  void SetPixelIndex(Int2 pixel_coord) { sequence->SetPixelIndex(pixel_coord); }
  void SetPointNum(int i) { sequence->SetPointNum(i); }
  void SetSubsequenceId(uint32_t id) { sequence->SetSubsequenceId(id); }

  // TODO: use the quasi-random sequence here.
  int UniformInt(int a, int b_inclusive) { return randgen.UniformInt(a, b_inclusive); }

  inline double Uniform01() 
  {
    return sequence->Uniform01();
  }

  inline Double2 UniformUnitSquare()
  {
    return sequence->UniformUnitSquare();
  }
};



template<class Iter>
void RandomShuffle(Iter begin, Iter end, RandGen &sampler)
{
  assert(begin != end);
	sampler.GetGenerator().shuffle(begin, end);
}


inline double PowerHeuristic(double prob_of_estimator_evaluated, std::initializer_list<double> other_probs)
{
  double sum = 0.;
  for (double p: other_probs)
    sum += p*p;
  double pp = Sqr(prob_of_estimator_evaluated);
  assert(sum+pp > 0.);
  // One of the densities should be positive. 
  // If something goes wrong and they are all zero, there is an added Epsilon to protect form returning NaN.
  // In this case, the path throughput better be zero or the result will be wrong!
  return pp/(pp+sum+Epsilon);
}


template<int n_, class T>
inline int TowerSampling(const T *probs, T r)
{
#ifndef NDEBUG
  for (int i=0; i<n_; ++i)
    assert(probs[i] >= 0.);
#endif
  // |- p0 -|- p1 -|- p2 -|- p3 -|
  //            r <--------------| // r falls in one of those bins.
  // Linear search. Measure r from the "rear".
  int n = n_-1;
  while (r >= probs[n] && n>0)
  {
    // Shed the last bin.
    r -= probs[n];
    --n;
  }
  assert(n >= 0 && n < n_);
  return n;
}


template<class T>
inline auto BisectionSearch(Span<const T> vals, T r)
{
  using index_t = typename decltype(vals)::index_t;
  static_assert(index_t(-1) < 0); // Should be signed type

  //         p0       (p0+p1)       (p0+p1+p2) ...                   (p0+...+pn-1==1)
  // |  i=0   |    i=1   |      i=2      |     ...   |        i=n-1          |
  //          0          1               2                                  n-1
  assert (vals.size() > 0);
  if (r < vals[0])
    return (index_t)0;
  if (r >= vals[vals.size()-1])
    return (index_t)vals.size();
  index_t first = 0;
  index_t last = (index_t)vals.size()-1;
  while (last > first+1)
  {
    index_t center = (first+last)/2;
    if  (r<vals[center])
      last = center;
    else
      first = center;
  }
  // r is now known to be between the values for first and last.
  return last;
};


template<class T>
inline auto TowerSamplingBisection(Span<const T> cmf, T r)
{
  auto upper = BisectionSearch(cmf, r);
  // E.g. if upper is zero, then r<p0, and we have to return upper to 
  // indicate that the random sample has fallen into the zero-th bin.
  upper = upper>=cmf.size() ? cmf.size()-1 : upper;
  // But because of roundoff errors, it may be that r>cmf[cmf.size()-1] which should ideally be one. 
  return upper;
}


template<class T>
inline void TowerSamplingInplaceCumSum(Span<T> weights)
{
  using index_t = decltype(weights.size());
  for (index_t i = 1; i < weights.size(); ++i)
  {
    weights[i] += weights[i - 1];
  }
}


template<class T>
inline void TowerSamplingNormalize(Span<T> cumsum)
{
  using index_t = decltype(cumsum.size());
  T inv_sum = T(1) / cumsum[cumsum.size() - 1];
  for (index_t i = 0; i < cumsum.size(); ++i)
  {
    cumsum[i] *= inv_sum;
  }
}


template<class T>
inline void TowerSamplingComputeNormalizedCumSum(Span<T> weights)
{
  TowerSamplingInplaceCumSum(weights);
  TowerSamplingNormalize(weights);
}


template<class T>
inline T TowerSamplingProbabilityFromCmf(Span<T> cmf, typename Span<T>::index_t idx)
{
  return cmf[idx] - (idx>0 ? cmf[idx-1] : T(0));
}


namespace OnlineVariance
{

// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
// sqr_deviation_sum = M_2,n
template<class T, class CounterType>
inline void Update(T& mean, T &sqr_deviation_sum, CounterType &num_previous_samples, const T &new_x)
{
  assert(num_previous_samples < std::numeric_limits<CounterType>::max());
  ++num_previous_samples;
  T delta = new_x - mean;
  mean = mean + delta / num_previous_samples;
  T delta2 = new_x - mean;
  sqr_deviation_sum += delta*delta2;
}

template<class T, class CounterType >
inline T FinalizeVariance(const T &sqr_deviation_sum, CounterType num_samples)
{
  if (num_samples < 2)
    return T{NaN};
  return sqr_deviation_sum/(num_samples-1);
}


}


namespace Accumulators
{

template<class T, class C = long>
class OnlineAverage
{
  T mean{};
  C counter{};
public:
  OnlineAverage(T initial_mean = T{}, C initial_count = C{})
    : mean{initial_mean}, counter{initial_count}
  {}

  const C Count() const {
    return counter;
  }

  void operator+=(const T &x)
  {
    ++counter;
    T delta = x - mean;
    mean = mean + delta / counter;
  }

  const T& operator()() const noexcept {
    return mean;
  }
};


template<class T, class C = long>
class SoaOnlineVariance;


// Convenience class, wrapping the algorithm.
template<class T, class CounterType_=int>
class OnlineVariance
{
  public:
    using CounterType = CounterType_;
    using ValueType = T;
    friend class SoaOnlineVariance<ValueType, CounterType>;
  private:
    T mean{0.};
    T sqr_deviation_sum{0.};
    CounterType n{0};
  public:
    OnlineVariance() = default;
    template<class U>
    explicit OnlineVariance(U zero) : mean{zero}, sqr_deviation_sum{zero} {}

    void Update(const T &new_x)
    {
      ::OnlineVariance::Update(mean, sqr_deviation_sum, n, new_x);
    }
    
    void Update(const OnlineVariance<T, CounterType> &other)
    {
      mean = (n + other.n > 0) ? (mean * static_cast<T>(n) + other.mean*static_cast<T>(other.n)) / (n + other.n) : T(0);
      n += other.n;
      sqr_deviation_sum += other.sqr_deviation_sum;
    }

    void operator+=(const T &new_x)
    {
      Update(new_x);
    }
    
    CounterType Count() const
    {
      return n;
    }
    
    T Mean() const 
    {
      return mean;
    }
    
    T Var() const
    {
      return ::OnlineVariance::FinalizeVariance(sqr_deviation_sum, n);
    }
    
    T Stddev() const
    {
      return sqrt(Var());
    }

    // Replace counts, keeping mean. Variance changes for small counts.
    // Not that the variance of the statistical mean value will change.
    template<class F>
    OnlineVariance<T,CounterType> CountsMultiplied(F c) const
    {
      OnlineVariance<T,CounterType> ret = *this;
      ret.n *= c;
      ret.sqr_deviation_sum *= c;
      return ret;
    }
};


template<class T, int rows, class C = long>
class OnlineCovariance
{
  using V = Eigen::Matrix<T, rows, 1>;
  using M = Eigen::Matrix<T, rows, rows>;
  V means{ Eigen::zero }; // w.r.t the offset
  V offset{ Eigen::zero }; // For more precision. Internal variables are expressed relative to this. Makes sense because points are expected to come as big clump far off the world origin.
  M xy_matrix{ Eigen::zero }; // sum (x - <x>)(y - <y>), i.e. sum over squared deviations from means
  C counter{};
public:
  void operator+=(const V &x)
  {
    if (unlikely(counter == 0))
    {
      offset = x;
    }
    // See, e.g. Schubert & Gertz (2018) "Numerically Stable Parallel Computation of (Co-)Variance"
    ++counter;
    V deltas_x = x - means - offset;
    means += deltas_x / counter;
    V deltas_y = x - means - offset; // Note: uses updated means.
    xy_matrix += deltas_x * deltas_y.transpose();
  }

  const C Count() const
  {
    return counter;
  }

  const V Mean() const
  {
    return means + offset;
  }

  const T Cov(int row, int col) const
  {
    return xy_matrix(row, col) / counter;
  }

  const M Cov() const
  {
    return xy_matrix / counter;
  }

  const V Var() const
  {
    return xy_matrix.diagonal() / counter;
  }
};


template<class T, class C>
class SoaOnlineVariance
{
public:
  using ArrayXd = Eigen::Array<T, Eigen::Dynamic, 1>;
  using ArrayXi = Eigen::Array<C, Eigen::Dynamic, 1>;
  SoaOnlineVariance(int size);
  void Add(int i, const T &new_x);
  void Add(const SoaOnlineVariance<T> &other);
  const auto &Mean() const { return mean; }
  const auto &Counts() const { return counts; }
  ArrayXd Var(T fill_value = NaN) const;
  ArrayXd MeanErr(T fill_value = NaN, C min_count = 2) const;
  int Size() const { return mean.rows(); }
  
  OnlineVariance<T,C> GetStats(int i) const
  {
    assert (i >= 0 && i<mean.rows());
    OnlineVariance<T,C> ret;
    ret.mean = mean[i];
    ret.sqr_deviation_sum = sqr_dev[i];
    ret.n = counts[i];
    return ret;
  }
  
  void SetStats(int i, const OnlineVariance<T, C> &ov)
  {
    assert (i >= 0 && i<mean.rows());
    mean[i] = ov.mean;
    sqr_dev[i] = ov.sqr_deviation_sum;
    counts[i] = ov.n;
  }

private:
  ArrayXd mean, sqr_dev;
  ArrayXi counts;
};


template<class T, class C>
inline SoaOnlineVariance<T,C>::SoaOnlineVariance(int size)
  : mean(size), sqr_dev(size), counts(size)
{
  mean.setZero();
  sqr_dev.setZero();
  counts.setZero();
}


template<class T, class C>
inline void SoaOnlineVariance<T,C>::Add(int i, const T & new_x)
{
  assert (i >= 0 && i<mean.rows());
  ::OnlineVariance::Update(mean[i], sqr_dev[i], counts[i], new_x);
}


template<class T, class C>
inline void SoaOnlineVariance<T,C>::Add(const SoaOnlineVariance<T>& other)
{
  mean = (counts + other.counts > 0).select(
    (mean * counts.template cast<T>() + other.mean * other.counts.template cast<T>()) / (counts + other.counts).template cast<T>(),
    0.
  );
  counts += other.counts;
  sqr_dev += other.sqr_dev;
}


template<class T, class C>
inline typename SoaOnlineVariance<T,C>::ArrayXd SoaOnlineVariance<T,C>::Var(T fill_value) const
{
  return (counts >= 2).select(sqr_dev / (counts - 1).template cast<T>(), ArrayXd::Constant(Size(),fill_value));
}

template<class T, class C>
inline typename SoaOnlineVariance<T,C>::ArrayXd SoaOnlineVariance<T,C>::MeanErr(T fill_value, C min_count) const
{
  return (counts >= min_count).select((sqr_dev.sqrt() / (counts - 1).template cast<T>()), ArrayXd::Constant(Size(),fill_value));
}


}



namespace PdfConversion
{

// TODO: Refactor so that it can be used in PT code. As of now used in Camera.
inline double AreaToSolidAngle(double segment_length, const Double3 &direction, const Double3 &normal)
{
  double result = Sqr(segment_length) / std::abs(Dot(direction, normal)+Epsilon);
  assert(result >= 0 && std::isfinite(result));
  return result;
}

// inline double SolidAngleToArea(double segment_length, const Double3 &direction, const Double3 &normal)
// {
//   double result = std::abs(Dot(direction, normal)) / (Sqr(segment_length)+Epsilon);
//   assert(result >= 0 && std::isfinite(result));
//   return result;
// }
// 
// // Area density is projected parallel to direction onto surface oriented according to normal.
// inline double ProjectArea(const Double3 &direction, const Double3 &normal)
// {
//   return std::abs(Dot(direction, normal));
// }

}



namespace SampleNamespace
{

// Use only for BSDF/Emitter samples which may come from delta distributions. Do not use for path densities.
// TODO: Maybe rename to ScatterPdf or something along those lines?
class Pdf
{
  double value;
public:
  operator double () const { return std::abs(value); }
  Pdf() noexcept : value{NaN} {}
  Pdf(double _value) noexcept : value{_value} 
  {
    assert(_value>=0);
  }
  Pdf(const Pdf &) noexcept = default;
  Pdf(Pdf &&) noexcept = default;
  Pdf& operator=(const Pdf &) noexcept = default;
  Pdf& operator=(Pdf &&) noexcept = default;

  bool IsFromDelta() const { return std::signbit(value); }  
  
  Pdf& operator *= (double q) 
  {
    assert(q >= 0);
    value *= q;
    return *this;
  }
  
  // Not permitted.
  friend Pdf operator*(Pdf a, Pdf b);
  
  friend Pdf operator*(double q, Pdf pdf)
  {
    pdf *= q;
    return pdf;
  }
  
  static Pdf MakeFromDelta(Pdf pdf) 
  {
    pdf.value = std::copysign(pdf.value, -1.);
    return pdf;
  }
};

inline double Value(const Pdf &pdf)
{
  return static_cast<double>(pdf);
}
  


template<class C, class U, class T>
struct Sample
{
  using CoordinateType = C;
  using ValueType = U;
  using Tag = T;
  
  CoordinateType coordinates;
  ValueType value; 
  Pdf pdf_or_pmf;
  
  template<class Other>
  auto as() const
  {
    return Sample<CoordinateType, ValueType, typename Other::Tag>{coordinates, value, pdf_or_pmf};
  }
  
  inline friend void SetPmfFlag(Sample &s) { s.pdf_or_pmf = Pdf::MakeFromDelta(s.pdf_or_pmf); }
  inline friend bool IsFromPmf(const Sample &s) { return s.pdf_or_pmf.IsFromDelta(); }
  inline friend bool IsFromPdf(const Sample &s) { return !IsFromPmf(s); }
  inline friend double PdfValue(const Sample &s) { assert(!IsFromPmf(s)); return (double)s.pdf_or_pmf; }
  inline friend double PmfValue(const Sample &s) { assert(IsFromPmf(s)); return (double)s.pdf_or_pmf; }
  inline friend double PmfOrPdfValue(const Sample &s) { return std::abs(s.pdf_or_pmf); }
};

}

using SampleNamespace::Pdf;
using SampleNamespace::Sample;

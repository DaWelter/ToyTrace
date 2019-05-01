#pragma once

#include "vec3f.hxx"
#include "util.hxx"
#include "span.hxx"

#include <cmath>
#include <random>
#include <array>
#include <algorithm>
#include <numeric>

namespace SampleTrafo
{
Double3 ToUniformDisc(Double2 r); // In the x-y plane
Double3 ToUniformSphere(Double2 r);
Double3 ToUniformHemisphere(Double2 r);
Double3 ToUniformSphereSection(double cos_opening_angle, Double2 r);
Double3 ToUniformSphereSection(Double2 r, double phi0, double z0, double phi1, double z1);
Double3 ToCosHemisphere(Double2 r);
Double3 ToPhongHemisphere(Double2 r, double alpha);
Double3 ToTriangleBarycentricCoords(Double2 r);
Double3 ToUniformSphere3d(const Double3 &r);
}


class Sampler
{
  std::mt19937 random_engine;
  std::uniform_real_distribution<double> uniform;
public:
  Sampler();
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
  
  static constexpr std::uint64_t default_seed = std::mt19937::default_seed;

  auto& GetGenerator()
  {
	  return random_engine;
  }
};


template<class Iter>
void RandomShuffle(Iter &&begin, Iter &&end, Sampler &sampler)
{
	std::shuffle(begin, end, sampler.GetGenerator());
  // Deprecated in c++14
  //std::random_shuffle(
  //  begin,
  //  end,
  //  [&sampler](int n) {
  //    return sampler.UniformInt(0, n-1);
  //  }
  //);
}


class Stratified2DSamples
{
    int nx, ny;
    inline void FastIncrementCoordinates()
    {
      // Avoid branching. Use faster conditional moves.
      ++current_x;
      current_y = (current_x >= nx) ? current_y+1 : current_y;
      current_y = (current_y >= ny) ? 0 : current_y;
      current_x = (current_x >= nx) ? 0 : current_x;
    }
  public:
    int current_x, current_y;
    
    Stratified2DSamples(int _nx, int _ny)
      : nx(_nx), ny(_ny),
        current_x(0), current_y(0)
      {}
    
    Double2 UniformUnitSquare(Double2 r)
    {
      Double2 ret{
        (current_x + r[0]) / nx,
        (current_y + r[1]) / ny
      };
      FastIncrementCoordinates();
      return ret;
    }
};


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
  return n;
}


template<class T>
inline auto BisectionSearch(Span<const T> vals, T r)
{
  using index_t = decltype(vals)::index_t;
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
inline void TowerSamplingComputeNormalizedCumSum(Span<T> weights)
{
  using index_t = decltype(weights.size());
  for (index_t i=1; i<weights.size(); ++i)
  {
    weights[i] += weights[i-1];
  }
  T inv_sum = T(1)/weights[weights.size()-1];
  for (index_t i=0; i<weights.size(); ++i)
  {
    weights[i] *= inv_sum;
  }
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
template<class T>
void Update(T& mean, T &sqr_deviation_sum, int &num_previous_samples, const T &new_x)
{
  ++num_previous_samples;
  T delta = new_x - mean;
  mean = mean + delta / num_previous_samples;
  T delta2 = new_x - mean;
  sqr_deviation_sum += delta*delta2;
}

template<class T>
T FinalizeVariance(const T &sqr_deviation_sum, int num_samples)
{
  if (num_samples < 2)
    return T{NaN};
  return sqr_deviation_sum/(num_samples-1);
}

// Convenience class, wrapping the algorithm.
template<class T>
class Accumulator
{
    T mean{0.};
    T sqr_deviation_sum{0.};
    int n{0};
  public:
    void Update(const T &new_x)
    {
      OnlineVariance::Update(mean, sqr_deviation_sum, n, new_x);
    }
    
    void operator+=(const T &new_x)
    {
      Update(new_x);
    }
    
    int Count() const
    {
      return n;
    }
    
    T Mean() const 
    {
      return mean;
    }
    
    T Var() const
    {
      return OnlineVariance::FinalizeVariance(sqr_deviation_sum, n);
    }
    
    T Stddev() const
    {
      return sqrt(Var());
    }
};


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
  Pdf() : value{NaN} {}
  Pdf(double _value) : value{_value} 
  {
    assert(_value>=0);
  }
  Pdf(const Pdf &other) : value{other.value} {}
  Pdf& operator=(const Pdf &other) { value = other.value; return *this; }
  
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

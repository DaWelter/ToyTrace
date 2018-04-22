#ifndef VEC3F_HXX
#define VEC3F_HXX

#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>

#include <boost/optional/optional.hpp>

template<class T, int d>
class Vec : public Eigen::Matrix<T, d, 1>
{
  public:
    typedef Eigen::Matrix<T, d, 1> Base;
    typedef T value_type;
    
    /*-------------------------------------------
     * ctors
     * -----------------------------------------*/
    
    Vec() : Base(Base::Constant(0.)) {}
    
    explicit Vec(const T &v) : Base() { this->setConstant(d, v); }
    
    Vec(const T &a, const T &b) : Base(a, b) {}
    Vec(const T &a, const T &b, const T &c) : Base(a, b, c) {}
    
    template<typename OtherDerived >
    Vec (const Eigen::MatrixBase< OtherDerived > &other) : Base(other) {}
};


#if 0
typedef Eigen::Matrix<int, 3, 1> Int3;
typedef Eigen::Matrix<int, 2, 1> Int2;
typedef Eigen::Matrix<double, 3, 1> Double3;
typedef Eigen::Matrix<double, 2, 1> Double2;
typedef Eigen::Matrix<double, 3, 1> Float3;
typedef Eigen::Matrix<double, 2, 1> Float2;
#define VECDARG class T,int d
#define VECD Eigen::Matrix<T, d, 1>
#else
typedef Vec<double, 3> Double3;
typedef Vec<double, 2> Double2;
typedef Vec<double, 3> Float3;
typedef Vec<double, 2> Float2;
#define VECDARG class T,int d
#define VECD Vec<T, d>
#endif

using Index3 = Eigen::Array<int, 3, 1>;

template<typename T>
using Matrix33 = Eigen::Matrix<T, 3, 3>;

template<class T>
constexpr int static_size()
{
  return 0;
}


// Don't need.
// namespace std
// {
//   template<class T, int dim>
//   struct hash<Vec<T,dim> >
//   {
//     std::size_t operator()(const Vec<T, dim> &key) const
//     {
//       std::size_t seed = 0;
//       for (int i=0; i<dim; ++i)
//         boost::hash_combine(seed, boost::hash_value(key[i]));
//       return seed;
//     }
//   };
// }

//namespace Eigen
//{
template<VECDARG>
inline std::ostream& operator<<(std::ostream &os, const VECD &v)
{
  os << "<";
  for (int i=0; i<d; ++i)
  {
    os << v[i];
    if (i < d -1) os << ",";
  }
  os << ">";
  return os;
}

template<VECDARG>
inline std::istream& operator>>(std::istream &is, VECD &v)
{
  char c;
  is >> c;
  if (c != '<') { is.setstate(std::ios_base::failbit); return is; }
  for (int i=0; i<d; ++i)
  {
    is >> v[i];
    if (i < d -1) is >> c;
    if (c != ',') { is.setstate(std::ios_base::failbit); return is; }
  }
  is >> c;
  if (c != '>') { is.setstate(std::ios_base::failbit); return is; }
  return is;
}


// template <class U, class T, int dim>
// inline Vec<U, dim> Cast(const Vec<T, dim> &v)
// {
//   return v.template cast<U>().eval();
// }

template<class T>
inline T Cross( const Eigen::Matrix<T,2,1>& u, const Eigen::Matrix<T,2,1> &v )
{
  return u.x()*v.y()-u.y()*v.x();
}

template<class Derived1, class Derived2>
auto Cross( const Eigen::MatrixBase<Derived1>& u, const Eigen::MatrixBase<Derived2> &v )
{
  return u.cross(v);
}

template<class Derived1, class Derived2>
auto Product( const Eigen::MatrixBase<Derived1>& u, const Eigen::MatrixBase<Derived2> &v )
{
  return u.cwiseProduct(v);
}

template<class Derived1, class Derived2>
auto Dot(const Eigen::MatrixBase<Derived1> &u, const Eigen::MatrixBase<Derived2> &v )
{
  return u.dot(v);
}

template<class Derived>
auto Length(const Eigen::MatrixBase<Derived> &a)
{
  return a.norm();
}

template<class Derived>
auto LengthSqr(const Eigen::MatrixBase<Derived> &a)
{
  return a.squaredNorm();
}

template<class Derived>
inline void Normalize(Eigen::MatrixBase<Derived>& u)
{
  return u.normalize();
}

template<class Derived>
auto Normalized(const Eigen::MatrixBase<Derived>& u)
{
  return u.normalized();
}

template<class Derived>
auto Reflected(const Eigen::MatrixBase<Derived>& reverse_incident_dir, const Eigen::MatrixBase<Derived>& normal)
{
  return (2.*reverse_incident_dir.dot(normal)*normal - reverse_incident_dir);
}


// Adapted from pbrt. eta is the ratio of refractive indices eta_i / eta_t
inline boost::optional<Double3> Refracted(const Double3 &wi, const Double3 &n, double eta_i_over_t) 
{
    const double eta = eta_i_over_t;
    double cosThetaI = Dot(n, wi);
    double sin2ThetaI = std::max(double(0), double(1 - cosThetaI * cosThetaI));
    double sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return boost::none;
    double cosThetaT = std::sqrt(1 - sin2ThetaT);
    double n_prefactor = (eta * std::abs(cosThetaI) - cosThetaT);
           n_prefactor = cosThetaI>=0. ? n_prefactor : -n_prefactor; // Invariance to normal flip.
    return Double3{-eta * wi + n_prefactor * n};
}


#ifndef NDEBUG
#define ASSERT_NORMALIZED(v) assert(std::abs(LengthSqr(v) - 1.) < 1.e-6)
#else
#define ASSERT_NORMALIZED(v) ((void)0)
#endif


template<class Scalar>
inline Scalar Clip(Scalar &x, Scalar a,Scalar b) {
  // TODO: Change this so that it preserve NaN input in x;
  if(x>b) x = b;
  else if(x<a) x = a;
  return x;
}


template<class Derived>
Eigen::Matrix<typename Eigen::internal::traits<Derived>::Scalar, 3,3> 
inline OrthogonalSystemZAligned(const Eigen::MatrixBase<Derived> &_Z)
{
  Eigen::Matrix<typename Eigen::internal::traits<Derived>::Scalar, 3 ,3> m;
  auto Z = m.col(2);
  auto X = m.col(0);
  auto Y = m.col(1);
  Z = _Z;
  ASSERT_NORMALIZED(Z);
  // Listing 3 in Duff et al. (2017) "Building an Orthonormal Basis, Revisited".
  float sign = std::copysignf(1.0f, Z[2]);
  const float a = -1.0f / (sign + Z[2]);
  const float b = Z[0] * Z[1] * a;
  X = Double3(1.0f + sign * Z[0] * Z[0] * a, sign * b, -sign * Z[0]);
  Y = Double3(b, sign + Z[1] * Z[1] * a, -Z[1]);
  return m;
}


template<class T, int N, int M>
inline Eigen::Array<T,M,1> Take(const Eigen::Array<T,N,1>& u, const Eigen::Array<int, M, 1> &indices)
{
  Eigen::Array<T,M,1> ret;
  for (int i=0; i<indices.size(); ++i)
  {
    assert(indices[i] >= 0 && indices[i]<u.size());
    ret[i] = u[indices[i]];
  }
  return ret;
}



constexpr auto Epsilon = std::numeric_limits<double>::epsilon();
constexpr auto Pi      = double(3.14159265358979323846264338327950288419716939937510);
constexpr auto Infinity= std::numeric_limits<double>::infinity();
constexpr auto LargeNumber = std::numeric_limits<double>::max()/16;
constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();
constexpr auto UnitSphereSurfaceArea = 4.*Pi;
constexpr auto UnitHalfSphereSurfaceArea = 2.*Pi;


#endif

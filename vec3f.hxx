#ifndef VEC3F_HXX
#define VEC3F_HXX

#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <boost/format.hpp>
#include <boost/functional/hash.hpp>

template<class T >
class DynArray;
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
typedef Eigen::Matrix<int, dim, 1> Intd;
typedef Eigen::Matrix<double, dim, 1> Doubled;
typedef Eigen::Matrix<int, 3, 1> Int3;
typedef Eigen::Matrix<int, 2, 1> Int2;
typedef Eigen::Matrix<double, 3, 1> Double3;
typedef Eigen::Matrix<double, 2, 1> Double2;
typedef Eigen::Matrix<double, 3, 1> Float3;
typedef Eigen::Matrix<double, 2, 1> Float2;
#define VECDARG class T,int d
#define VECD Eigen::Matrix<T, d, 1>
#else
//typedef Vec<int, dim> Intd;
//typedef Vec<double, dim> Doubled;
typedef Vec<int, 3> Int3;
typedef Vec<int, 2> Int2;
typedef Vec<bool, 3> Bool3;
typedef Vec<bool, 2> Bool2;
typedef Vec<double, 3> Double3;
typedef Vec<double, 2> Double2;
typedef Vec<double, 3> Float3;
typedef Vec<double, 2> Float2;
typedef Vec<long, 2> Long2;
typedef Vec<long, 2> Long3;
#define VECDARG class T,int d
#define VECD Vec<T, d>
#endif

namespace std
{
  template<class T, int dim>
  struct hash<Vec<T,dim> >
  {
    std::size_t operator()(const Vec<T, dim> &key) const
    {
      std::size_t seed = 0;
      for (int i=0; i<dim; ++i)
        boost::hash_combine(seed, boost::hash_value(key[i]));
      return seed;
    }
  };
}

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


template <class U, class T, int dim>
inline Vec<U, dim> Cast(const Vec<T, dim> &v)
{
  return v.template cast<U>().eval();
}

template<class T>
inline T Cross( const Vec<T,2>& u, const Vec<T,2> &v )
{
  return u.x()*v.y()-u.y()*v.x();
}

template<class Derived1, class Derived2>
inline typename Eigen::internal::plain_matrix_type<Derived1>::type Cross( const Eigen::MatrixBase<Derived1>& u, const Eigen::MatrixBase<Derived2> &v )
{
  return u.cross(v);
}

template<class Derived1, class Derived2>
inline typename Eigen::internal::plain_matrix_type<Derived1>::type Product( const Eigen::MatrixBase<Derived1>& u, const Eigen::MatrixBase<Derived2> &v )
{
  return u.cwiseProduct(v);
}

template<class Derived1, class Derived2>
inline typename Eigen::internal::traits<Derived1>::Scalar Dot(const Eigen::MatrixBase<Derived1> &u, const Eigen::MatrixBase<Derived2> &v )
{
  return u.dot(v);
}

template<class Derived>
inline typename Eigen::internal::traits<Derived>::Scalar Length(const Eigen::MatrixBase<Derived> &a)
{
  return a.norm();
}

template<class T>
inline void Normalize(Vec<T,3>& u)
{
  return u.normalize();
}

template<class Scalar>
inline Scalar Clip(Scalar &x, Scalar a,Scalar b){
  if(x>b) x = b;
  else if(x<a) x = a;
  return x;
}


constexpr auto Epsilon = 1.e-8; //std::numeric_limits<double>::epsilon();
constexpr auto Pi      = double(3.14159265358979323846264338327950288419716939937510);
constexpr auto Infinity= std::numeric_limits<double>::infinity();

#endif

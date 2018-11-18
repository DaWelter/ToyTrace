#pragma once

#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>
#include <unordered_map>

#include <boost/align/aligned_allocator.hpp>

template<class T>
inline T Sqr(const T &x)
{
  return x*x;
}

template<class T>
inline T Heaviside(const T &x)
{
  return x>T{} ? T{1} : T{0};
}


inline double Rcp(double x)
{
  return 1./x;
}

inline float Rcp(float x)
{
  return 1.f/x;
}


// Note: Will happily take the signbit from zero. So the result for 0 is basically random.
template<class T, typename std::enable_if_t<std::is_floating_point<T>{}, int> = 0>
inline T Sign(const T &x)
{
  return std::copysign(1., x);
}


// Also from PBRT. Used to compute error bounds for floating point arithmetic. See pg. 216.
template<class T>
inline constexpr T Gamma(int n) {
    constexpr T eps_half = std::numeric_limits<T>::epsilon();
    return (n * eps_half) / (1 - n * eps_half);
}


template<class T>
inline bool Quadratic(T a, T b, T c, float &t0, float &t1)
{
  //from PBRT pg. 1080
  const T d = b*b - T(4)*a*c;
  if (d < T(0))
    return false;
  const T sd = std::sqrt(d);
  const T q = b<0 ? -b+sd : -b-sd;
  t0 = q/T(2)/a;
  t1 = T(2)*c/q;
  if (t0 > t1)
    std::swap(t0 ,t1);
  return true;
}


namespace quadratic_internal
{

template<class T>
inline T Errorformula1(T A, T B, T C, T D, T sD, T eA, T eB, T eC)
{
  constexpr T eps = std::numeric_limits<T>::epsilon();
  const T xi = B<T(0) ? T(1) : T(-1);
  const T G = B-xi*sD;
  const T Ainv = T(1)/A;
  const T sDinv = T(1)/sD;
  const T E1 = std::abs(G*Ainv) + T(3)/T(4)*std::abs(sD*Ainv) + std::abs(C*sDinv) + std::abs(B*B*Ainv*sDinv)/T(4);
  const T E2 = eA*std::abs((C*xi*Ainv*sDinv - G*Ainv*Ainv/T(2))) + eB/T(2)*std::abs(Ainv*(B*xi*sDinv-T(1))) + eC*std::abs(sDinv);
  return eps*E1 + E2;
}

template<class T>
inline T Errorformula2(T A, T B, T C, T D, T sD, T eA, T eB, T eC)
{
  constexpr T eps = std::numeric_limits<T>::epsilon();
  const T xi = B<T(0) ? T(1) : T(-1);
  const T G = B - xi*sD;
  const T sDinv = T(1)/sD;
  const T GGsDinv = T(1)/(G*G*sD);
  const T E1 = std::abs(GGsDinv)*(T(4)*std::abs(C*G*sD) + T(3)*std::abs(C*D) + T(4)*std::abs(A*C*C) + std::abs(B*B*C));
  const T E2 = std::abs(GGsDinv)*(eA*T(4)*std::abs(C*C)+ T(2)*eB*std::abs(C*(B*xi - sD)) + eC*std::abs((T(4)*A*C*xi - T(2)*G*sD)));
  return eps*E1 + E2;
}

};


template<class T>
inline bool Quadratic(T a, T b, T c, T ea, T eb, T ec, T &t0, T &t1, T &err0, T &err1)
{
  using namespace quadratic_internal;
  //from PBRT pg. 1080
  const T d = b*b - T(4)*a*c;
  if (d < T(0))
    return false;
  const T sd = std::sqrt(d);
  err0 = Errorformula1(a, b, c, d, sd, ea, eb, ec);
  err1 = Errorformula2(a, b, c, d, sd, ea, eb, ec);
  const T q = b<0 ? -b+sd : -b-sd;
  t0 = q/T(2)/a;
  t1 = T(2)*c/q;
  if (t0 > t1)
  {
    std::swap(t0 ,t1);
    std::swap(err0, err1);
  }
  return true;
}


namespace strconcat_internal
{

/* Terminate the recursion. */
inline void impl(std::stringstream &ss)
{
}
/* Concatenate arbitrary objects recursively into a string using a stringstream.
   Adapted from https://en.wikipedia.org/wiki/Variadic_template
*/
template<class T, class ... Args>
inline void impl(std::stringstream &ss, const T& x, Args&& ... args)
{
  ss << x;
  impl(ss, args...);
}

}

/* A simple, safe replacement for sprintf with automatic memory management.
*/
template<class ... Args>
inline std::string strconcat(Args&& ...args)
{
  std::stringstream ss;
  strconcat_internal::impl(ss, args...);
  return ss.str();
}



namespace strformat_internal
{

namespace
{

#pragma GCC diagnostic push
// The compiler complains that this function is not used. But it really is or else my 
// strformat would not compile at all because the compile time(!) recursion could not be terminated.
#pragma GCC diagnostic ignored "-Wunused-function"  // GCC does not even care ...
  
// Terminate recursion. All arguments were sunk.
void impl(std::stringstream &ss, const std::string &format, int start)
{
  auto pos = format.find('%', start);
  
  if (pos != std::string::npos)
    throw std::invalid_argument("Too few arguments for strformat");
  
  ss << format.substr(start, format.size()-start);
}

#pragma GCC diagnostic pop // Restore command line options.


template<class T, class ... Args>
void impl(std::stringstream &ss, const std::string &format, int start, const T& x, Args&& ... args)
{
  // Note: "for a non-empty substring, if pos >= size(), the function always returns npos."
  // Ref: http://en.cppreference.com/w/cpp/string/basic_string/find
  auto pos = format.find('%', start);
  
  if (pos == std::string::npos)
    throw std::invalid_argument("Too many arguments for strformat");
  
  ss << format.substr(start, pos-start);
  
  if (pos < format.size()-1 && format[pos+1]=='%') // Escaped %% 
  {
    ss << '%';
    impl(ss, format, pos+2, x, args...);
    return;
  }      
  
  ss << x;
  
  impl(ss, format, pos+1, args...);
}

}
  
};


template<class ... Args>
inline std::string strformat(const std::string &format, Args&& ...args)
{
  std::stringstream ss;
  strformat_internal::impl(ss, format, 0, args...);
  return ss.str();
}




inline bool startswith(const std::string &a, const std::string &b)
{
  if (a.size() < b.size())
    return false;
  return a.substr(0, b.size()) == b;
}


// Copy & Paste Fu! https://stackoverflow.com/questions/27140778/range-based-for-with-pairiterator-iterator
template <typename I>
struct iter_pair : std::pair<I, I>
{ 
    using std::pair<I, I>::pair;

    I begin() { return this->first; }
    I end() { return this->second; }
};


// std::vector with 16 byte aligment as required by Eigen's fixed size types.
// This class also comes with range checking in debug mode.
template<class T, class Alloc = boost::alignment::aligned_allocator<T, 16>>
class ToyVector : public std::vector<T, Alloc>
{
  using B = std::vector<T, Alloc>;
public:
  using B::B;
  
  inline typename B::const_reference operator[](typename B::size_type i) const
  {
    assert(i >= 0 && i<B::size());
    return B::operator[](i);
  }
  
  inline typename B::reference operator[](typename B::size_type i)
  {
    assert(i >= 0 && i<B::size());
    return B::operator[](i);
  }
};


template<class K, class T, class F, class Hash, class Pred>
inline T GetOrInsertFromFactory(std::unordered_map<K, T, Hash, Pred> &m, const K &k, F factory)
{
  auto it = m.find(k);
  if (it == m.end())
  {
    auto& t = m[k] = factory();
    return t;
  }
  else
    return it->second;
}


template<class T>
inline T ASSERT_NOT_NULL(T x, typename std::enable_if<std::is_pointer<T>::value>::type* = 0)
{
  assert(x != nullptr);
  return x;
}


// Adapted from http://the-witness.net/news/2012/11/scopeexit-in-c11/
template <typename F>
struct ScopeExit {
    ScopeExit(F f) : f(f) {}
    ~ScopeExit() { f(); }
    F f;
};

template <typename F>
ScopeExit<F> MakeScopeExit(F f) {
    return ScopeExit<F>(f);
};

#define SCOPE_EXIT(code) \
    auto scope_exit_ ## __LINE__ = MakeScopeExit([=](){ code })

    
inline int RowMajorOffset(int x, int y, int size_x, int size_y)
{
  return x + y*size_x;
}

inline std::pair<int, int> RowMajorPixel(int offset, int size_x, int size_y)
{
  int y = offset / size_x;
  int x = offset - y*size_x;
  return std::make_pair(x,y);
}

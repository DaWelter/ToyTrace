#pragma once

#include <iostream>
#include <cassert>
#include <limits>
#include <boost/align/aligned_allocator.hpp>

template<class T>
inline T Sqr(const T &x)
{
  return x*x;
}


// Straight from PBRT
static constexpr double MachineEpsilon =
  std::numeric_limits<double>::epsilon() * 0.5;

// Also from PBRT. Used to compute error bounds for floating point arithmetic. See pg. 216.
inline constexpr double Gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
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

/* A simple, safe, replacement for sprintf with automatic memory management.
   I will probably implement this for String, too.
*/
template<class ... Args>
std::string strconcat(Args&& ...args)
{
  std::stringstream ss;
  strconcat_internal::impl(ss, args...);
  return ss.str();
}



#if 0 // if using c++11
namespace std {
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
}
#endif


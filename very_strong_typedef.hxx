#pragma once

#include <cmath>
#include <iostream>
#include <boost/config.hpp>
#include <boost/operators.hpp>

// Taking strong inspiration from the answers on stackoverflow.
// https://stackoverflow.com/questions/23726038/how-can-i-create-a-new-primitive-type-using-c11-style-strong-typedefs
// as well as from Boost strong typedef
// http://www.boost.org/doc/libs/1_61_0/libs/serialization/doc/strong_typedef.html
// In contrast to the boost version I don't want automatic conversion from T to the new strong typedef'ed type.
// This is uselful in case I want to use float as base type. In this case I use c++11 custom literals that
// return the appropriate strong type. That has the benefit of avoiding inefficiencies due to unwanted
// double to float conversion if I forget to write 1.f instead of 1.
// Moreover I don't want the strong type to convert to the base type. Otherwise function applications like
// strong_type x{5.}; pow(x); could implicitly convert to the base type (??).
template <typename T, typename N>
struct very_strong_typedef : public \
  boost::field_operators1<very_strong_typedef<T,N>,
  boost::totally_ordered1<very_strong_typedef<T,N>, 
  boost::totally_ordered2<very_strong_typedef<T,N>,T>>>
{
    using strong_type = very_strong_typedef<T,N>;
    using type = T; // the wrapped type
    T value_;

    // Constexpr should make that construction with a literal argument 
    // leads to conversion to the appropriate base type at compile time.
    constexpr explicit very_strong_typedef(T val): value_(val){};
    very_strong_typedef(){value_={};}; // default, zero-initialization

    // Conversion operators. But I don't want auto conversion.
    // operator T& () { return value_; }
    // operator const T& () const { return value_; }

    // These are address of operators.
    T* operator &() { return &value_; }
    const T* operator& () const { return &value_; }
     
    strong_type& operator+=(const strong_type& rhs)
    {
        value_+=rhs.value_; 
        return *this;
    }
    
    strong_type& operator*=(const strong_type& rhs)
    {
      value_*=rhs.value_;
      return *this;
    }

    strong_type& operator/=(const strong_type& rhs)
    {
      value_/=rhs.value_;
      return *this;
    }
    
    strong_type& operator-=(const strong_type& rhs)
    {
      value_-=rhs.value_;
      return *this;
    }
    
    bool operator<(const strong_type& rhs) const
    {
      return value_ < rhs.value_;
    }

    // Function definitions for use with argument dependent lookup
    // http://en.cppreference.com/w/cpp/language/adl
    friend inline type value(strong_type x)
    {
      return x.value_;
    }
    
    
    friend inline strong_type pow(strong_type x, strong_type e) 
    { 
      return strong_type{std::pow(value(x), value(e))}; 
    }
    
    friend inline strong_type abs(strong_type x)
    {
      using namespace std;
      return strong_type{abs(x.value_)};
    }
    
    friend inline auto isfinite(strong_type x)
    {
      using namespace std;
      return isfinite(value(x));
    }
    
    friend std::ostream& operator<<(std::ostream & lhs, const strong_type& rhs)
    {
        lhs << rhs.value_;
        return lhs;
    }
};


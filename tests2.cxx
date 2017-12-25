#include <iostream>
#include <boost/serialization/strong_typedef.hpp>

#include "gtest/gtest.h"
#include "spectral.hxx"
#include "vec3f.hxx"

void f1(Color::RGBScalar q)
{
}

void f2(const Color::RGBScalar &q)
{
}

void f3(double)
{
}

double f4(const double &x)
{
  return x;
}


BOOST_STRONG_TYPEDEF(double, BoostType);


TEST(StrongTypedef, ScalarOperations)
{
  Color::RGBScalar a{1.};
  Color::RGBScalar b{2.};
  std::cout << a*b << std::endl;
  
  //f1(a+1.); // Error
  //f2(a+1.); // Error
  
  f3(value(a));
  
  std::cout << a+1._rgb << std::endl;
  Color::RGBScalar x = 2._rgb;
  x = pow(a+3._rgb, 2._rgb);
  
  Color::RGBScalar c{3.};
  std::cout << a+c << std::endl;
  Color::RGBScalar d = a+c;
  std::cout << d << std::endl;
};


TEST(StrongTypedef, ArgumentDependentLookup)
{
  using namespace Color;
  RGBScalar x{-1.1};
  RGBScalar absx = abs(x); // Is the integer version of abs being called?
  ASSERT_EQ(value(absx), 1.1);
}


TEST(StrongTypedef, EigenSupport)
{
  using namespace Color;
  RGB a{1._rgb, 2._rgb, 3._rgb};
  RGB b{4._rgb, 5._rgb, 6._rgb};
  RGB c = a * b;
  RGB d = a / c;
}


TEST(StrongTypedef, Boost1)
{
  // I don't want this to compile because f4 and f3 take double arguments, not BoostType!
  BoostType x{1.};
  double y = f4(x);
  f3(x);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
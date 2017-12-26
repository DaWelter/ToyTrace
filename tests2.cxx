#include <iostream>
#include <boost/serialization/strong_typedef.hpp>
#include <boost/pool/simple_segregated_storage.hpp>

#include "gtest/gtest.h"
#include "spectral.hxx"
#include "vec3f.hxx"

namespace StrongTypedefDetail
{

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

} //StrongTypedefDetail

TEST(StrongTypedef, ScalarOperations)
{
  using namespace StrongTypedefDetail;
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
  using namespace StrongTypedefDetail;
  using namespace Color;
  RGBScalar x{-1.1};
  RGBScalar absx = abs(x); // Is the integer version of abs being called?
  ASSERT_EQ(value(absx), 1.1);
}


TEST(StrongTypedef, EigenSupport)
{
  using namespace StrongTypedefDetail;
  using namespace Color;
  RGB a{1._rgb, 2._rgb, 3._rgb};
  RGB b{4._rgb, 5._rgb, 6._rgb};
  RGB c = a * b;
  RGB d = a / c;
}


TEST(StrongTypedef, Boost1)
{
  using namespace StrongTypedefDetail;
  BOOST_STRONG_TYPEDEF(double, BoostType);
  // I don't want this to compile because f4 and f3 take double arguments, not BoostType!
  BoostType x{1.};
  double y = f4(x);
  f3(x);
}




namespace SmallObjectStorageDetail
{

using std::ostream;
using std::cout;
using std::endl;

class Base
{
public:
  Base() : a{'a'} {}
  virtual ~Base() {}
  char a;
  virtual ostream& print(ostream &os) const 
  {
    os << "Base@" << this << "(" << a << ")"; return os;
  }
};

class Derived1 : public Base
{
public:
  Derived1() : b{1.} {}
  double b;
  virtual ostream& print(ostream &os) const override
  {
    os << "Derived1@" << this << "(" << a << " " << b << ")"; return os;
  }
};

class Derived2 : public Base
{
public:
  Derived2() : c{2.}, d{3.} {}
  double c;
  double d;
  virtual ostream& print(ostream &os) const override
  {
    os << "Derived2@" << this << "(" << a << " " << c << " " << d << ")"; return os;
  }
};

}


TEST(SmallObjectStorage, PolymorphicClasses)
{
  using namespace SmallObjectStorageDetail;
  boost::simple_segregated_storage<std::size_t> storage;
  std::vector<char> v(1024);
  constexpr std::size_t max_elem_size = 
    std::max(sizeof(Base),
    std::max(sizeof(Derived1),
             sizeof(Derived2)));
  storage.add_block(&v.front(), v.size(), max_elem_size);
  cout << "element size = " << max_elem_size << " bytes" << endl;
  Base* derived1 = new (storage.malloc()) Derived1();
  Base* derived2 = new (storage.malloc()) Derived2();
  Base* derived1b = new (storage.malloc()) Derived1();
  EXPECT_GE((derived2-derived1)*sizeof(Base), max_elem_size);
  EXPECT_GE((derived1b-derived2)*sizeof(Base), max_elem_size);
  cout << "1 = "; derived1->print(cout); cout << endl;
  cout << "2 = "; derived2->print(cout); cout << endl;
  cout << "3 = "; derived1b->print(cout); cout << endl;
  EXPECT_EQ(((std::size_t)&derived1) % sizeof(double), 0); // Alignment
}



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
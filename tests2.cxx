#include <iostream>
#include <typeinfo>
#include <bitset>
#include <type_traits>
#include <unordered_map>

#include <boost/serialization/strong_typedef.hpp>
#include <boost/pool/simple_segregated_storage.hpp>
#include <boost/variant.hpp>
#include <boost/variant/polymorphic_get.hpp>
#include <boost/align/aligned_allocator.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/any.hpp>

#include "gtest/gtest.h"
#include "spectral.hxx"
#include "vec3f.hxx"
#include "util.hxx"
#include "very_strong_typedef.hxx"


TEST(BasicAssumptions, InheritCtor)
{
  struct A
  {
    int a;
    A(int a) : a{a} {}
  };
  
  struct B : public A
  {
    using A::A;
    int f() const { return this->a; }
  };
  
  EXPECT_EQ(B(42).f(), 42);
}


TEST(BasicAssumptions, ToyVector)
{
  ToyVector<double> fu { 1, 2 };
  std::cout << typeid(fu).name() << " accessing elem 1: " << fu[1] << std::endl;
#ifndef NDEBUG
  ASSERT_DEATH(fu[2], ".*Assertion.*");
#endif
}


namespace UnionTestDetail
{

struct NoDefaultCtor
{
  int a;
  explicit NoDefaultCtor(int _a) : a{_a} {}
  NoDefaultCtor() = delete;
};
  
  
struct UnionTesting
{
  union U
  {
    NoDefaultCtor ctor;
    double d;
    U() {}
  };
  U u;
  UnionTesting() 
  {
    u.ctor = NoDefaultCtor{7};
  }
  UnionTesting(double _d)
    {
      u.d = _d;
    }
};
  
};


TEST(BasicAssumptions, Unions)
{
  UnionTestDetail::UnionTesting t1{};
  UnionTestDetail::UnionTesting t2{0.5};
  std::cout << t2.u.d << std::endl;
}


namespace EnableIfTestDetail
{

struct Base
{
  int a = { 7 };
};

struct Derived : public Base
{
  int b = { 42 };
};


struct EnableIfOperatorTest
{
  template<class T, typename std::enable_if_t<std::is_base_of<Base,T>{}, int> = 0>
  bool operator()(const T& u) const
  {
    return true;
  }
  
  template<class T, typename std::enable_if_t<!std::is_base_of<Base,T>{}, int> = 0>
  bool operator()(const T& u) const
  {
    return false; // Default
  }

  // Compile error! :-(
//   template<class T>
//   bool operator()(const T& u) const
//   {
//     return false; // Default
//   }
};

}

TEST(BasicAssumptions, EnableIf)
{
  using namespace EnableIfTestDetail;
  Base base;
  Derived derived;
  int a_int = 9;
  EXPECT_FALSE(EnableIfOperatorTest()(a_int));
  EXPECT_TRUE(EnableIfOperatorTest()(base));
  EXPECT_TRUE(EnableIfOperatorTest()(derived));
}


TEST(BasicAssumptions, BitsetSize)
{
  static constexpr int NBITS = 128;
  std::bitset<NBITS> bs;
  EXPECT_EQ(sizeof(bs), NBITS/8);
}


struct RVODemo
{
  volatile int q; 
  // Volatile, hoping that it will stop the compiler from replacing my code with values computed at compile time.
  // I suppose it could do that since the inputs are literals. 
  // And the code is trivial. Just a branch and a multiplication, essentially.
  int &num_copies;
  RVODemo(volatile int q, int &_num_copies) : q{q}, num_copies{_num_copies} {}
  RVODemo(const RVODemo &demo) 
    : q{demo.q}, num_copies{demo.num_copies} 
    { ++num_copies; }
  RVODemo& operator=(const RVODemo &other)
  {
    ++num_copies;
    q = other.q;
    return *this;
  }
};

RVODemo RVODemoFunc(volatile int q, int &num_copies)
{
  if (q == 1)
  {
    return RVODemo{q, num_copies};
  }
  else
  {
    return RVODemo{q*q, num_copies};
  }
};

RVODemo RVODemoFunc2(volatile int q, int &num_copies)
{
  RVODemo result{0, num_copies};
  if (q == 1)
  {
    result.q = q;
  }
  else
  {
    result.q = q*q;
  }
  return result;
};


RVODemo RVODemoFunc3(volatile int q, int &num_copies)
{
  RVODemo result{q, num_copies};
  if (q == 1)
  {
    return RVODemo{1, num_copies};
  }
  result.q = q*q;
  return result; // <-- copy!
};


std::tuple<RVODemo, int> RVODemoFunc4(volatile int q, int &num_copies)
{
  if (q == 1)
  {
    return std::make_tuple(RVODemo{q, num_copies}, 42);
  }
  else
  {
    return std::make_tuple(RVODemo{q*q, num_copies}, 7);
  }  
}


TEST(BasicAssumptions, RVO)
{
  int num_copies;
  volatile int q = 1;
  {
    num_copies = 0;
    RVODemo demo = RVODemoFunc(q, num_copies);
    std::cout << "q=" << demo.q << " num_copies =" << num_copies << std::endl;
    EXPECT_EQ(num_copies, 0);
  }
  {
    num_copies = 0;
    RVODemo demo = RVODemoFunc2(q, num_copies);
    std::cout << "q=" << demo.q << " num_copies =" << num_copies << std::endl; 
    EXPECT_EQ(num_copies, 0);
  }
  {
    num_copies = 0;
    RVODemo demo = RVODemoFunc3(q, num_copies);
    std::cout << "q=" << demo.q << " num_copies =" << num_copies << std::endl;
    EXPECT_EQ(num_copies, 0);
  }
  { 
    num_copies = 0;
    RVODemo demo{0, num_copies};
    int demo_other = 0;
    std::tie(demo, demo_other) = RVODemoFunc4(q, num_copies);
    std::cout << "Demo4: q=" << demo.q << " num_copies =" << num_copies << std::endl;
    EXPECT_EQ(num_copies, 2); // Initial construction of tuple + copy into destination variable via std::tie.
  }
  q = 2;
  {
    num_copies = 0;
    RVODemo demo = RVODemoFunc(q, num_copies);
    std::cout << "q=" << demo.q << " num_copies =" << num_copies << std::endl;
    EXPECT_EQ(num_copies, 0);
  }
  {
    num_copies = 0;
    RVODemo demo = RVODemoFunc2(q, num_copies);
    std::cout << "q=" << demo.q << " num_copies =" << num_copies << std::endl;
    EXPECT_EQ(num_copies, 0);
  }
  { 
    num_copies = 0;
    RVODemo demo = RVODemoFunc3(q, num_copies);
    std::cout << "q=" << demo.q << " num_copies =" << num_copies << std::endl;
    EXPECT_EQ(num_copies, 1);
  }
}



struct Stuff
{
  double a;
  char b;
  char c;
};
static_assert(sizeof(Stuff) == 16, "Unexpected size!");
struct MoreStuff
{
  Stuff s;
  char c;
};
static_assert(sizeof(MoreStuff) == 24, "Unexpected size!");


TEST(BasicAssumptions, FloatType)
{
  double nan = NaN;
  double inf = Infinity;
  EXPECT_FALSE(nan == nan);
  EXPECT_TRUE(nan != nan);
  EXPECT_TRUE(std::isnan(nan));
  EXPECT_FALSE(nan < 0.);
  EXPECT_FALSE(nan >= 0.);
  EXPECT_TRUE(inf > 1.);
  EXPECT_TRUE(-inf < -1.);
  EXPECT_TRUE(inf == inf);
  EXPECT_FALSE(inf > inf);
  EXPECT_TRUE(inf >= inf);
  EXPECT_FALSE(inf == -inf);
  EXPECT_TRUE(-inf == -inf);
  EXPECT_TRUE(inf+1. == inf);
  EXPECT_TRUE(std::isnan(inf/(inf+1.)));
}


TEST(BasicAssumptions, EigenTypes)
{
  // Spectral3 is currently an Eigen::Array type. It is still a row vector/array.
  EXPECT_EQ(Spectral3::ColsAtCompileTime, 1);
  EXPECT_EQ(Spectral3::RowsAtCompileTime, 3);
  // Vectors in eigen are Eigen::Matrix row vectors.
  EXPECT_EQ(Double3::ColsAtCompileTime, 1);
  EXPECT_EQ(Double3::RowsAtCompileTime, 3);
};


TEST(BasicAssumptions, CommaInitializers)
{
  Eigen::Matrix<float, 3, 2> m;
  using M31 = Eigen::Matrix<float, 3, 1>;
  using M12 = Eigen::Matrix<float, 1, 2>;
  using M21 = Eigen::Matrix<float, 2, 1>;
  M31 v; v << 1, 2, 3;
  m << v, v;
  std::cout << "-- col by col fill ---" << std::endl;
  std::cout << m << std::endl;
  M12 w; w << 5, 6;
  m << w, w, w;
  std::cout << "-- row by row fill ---" << std::endl;
  std::cout << m << std::endl;
  M21 s; s << 8, 9;
  ///m << s, s, s, s; // Not working because it tries to stack the vectors vertically. So the number of components do not match.
  auto r = s.transpose();
  m << r, r, r;
  std::cout << "-- row by row fill (transposed vec) ---" << std::endl;
  std::cout << m << std::endl;
  m.transpose() << s, s, s;
  std::cout << "-- row by row fill (transposed dst) ---" << std::endl;
  std::cout << m << std::endl;
}


TEST(BasicAssumptions, AlignmentAllocator)
{
  std::vector<double, boost::alignment::aligned_allocator<double, 128>> v{1., 2., 3.};
  EXPECT_EQ(((std::size_t)&v[0]) % 128, 0);
}


TEST(BasicAssumptions, UniqueAlgo)
{
  std::vector<int> elems{1, 2, 2, 3, 3, 4};
  auto it = std::unique(elems.begin(), elems.end());
  elems.resize(it - elems.begin());
  EXPECT_GE(elems.capacity(), 6);
  EXPECT_EQ(elems.size(), 4);
}


TEST(BasicAssumptions, NewMax)
{
  constexpr int a = std::max({1, 2, 3, 4, 3, 2, 1});
  static_assert(a == 4, "Must be the maximum");
}


template<class Derived1, class Derived2>
inline auto TestExpression( const Eigen::MatrixBase<Derived1>& u, const Eigen::MatrixBase<Derived2> &v )
{
  return (u.array()*v.array()).matrix();
}

// Because auto does not work well with expression templates.
// See https://eigen.tuxfamily.org/dox/TopicPitfalls.html

// These tests can identify an returned expression structure by
// clearing the memory of the arguments before the expression is
// evaluated!
#define UGLY_EXPR_TEST_BINARY_VEC_RETURN(func, x0, x1, y0, y1) \
  Double3 a{x0, x1, 0}, b{y0, y1, 0}; \
  auto d = func(a, b);  \
  a = Double3{0}; \
  b = Double3{0}; \
  Double3 result{d};
  
#define UGLY_EXPR_TEST_BINARY_DOUBLE_RETURN(func, x0, x1, y0, y1) \
  Double3 a{x0, x1, 0}, b{y0, y1, 0}; \
  auto d = func(a, b);  \
  a = Double3{0}; \
  b = Double3{0}; \
  double result = d;

#define UGLY_EXPR_TEST_UNARY_DOUBLE_RETURN(func, x0, x1) \
  Double3 a{x0, x1, 0}; \
  auto d = func(a);  \
  a = Double3{0}; \
  double result = d;

#define UGLY_EXPR_TEST_UNARY_VEC_RETURN(func, x0, x1) \
  Double3 a{x0, x1, 0}; \
  auto d = func(a);  \
  a = Double3{0}; \
  Double3 result{d};
  
 
TEST(BasicAssumptions, ExpressionTemplates1)
{
  UGLY_EXPR_TEST_BINARY_VEC_RETURN(TestExpression, 1, 0, 1, 0)
  ASSERT_NEAR(result[0], 0., 1.e-3);
}


TEST(BasicAssumptions, ExpressionTemplates2)
{
  UGLY_EXPR_TEST_BINARY_VEC_RETURN(Cross, 1, 0, 0, 1)
  ASSERT_NEAR(result[2], 1., 1.e-3);
}


TEST(BasicAssumptions, ExpressionTemplates3)
{
  UGLY_EXPR_TEST_BINARY_VEC_RETURN(Product, 2, 2, 3, 3)
  ASSERT_NEAR(result[0], 0., 1.e-3); // Would be 6 but since the return is an expression type, we expect 0.
}


TEST(BasicAssumptions, ExpressionTemplates4)
{
  UGLY_EXPR_TEST_BINARY_DOUBLE_RETURN(Dot, 2, 2, 1, 0)
  ASSERT_NEAR(result, 2., 1.e-3);
}


TEST(BasicAssumptions, ExpressionTemplates5)
{
  UGLY_EXPR_TEST_UNARY_DOUBLE_RETURN(Length, 2, 2)
  ASSERT_NEAR(result, std::sqrt(8), 1.e-3);
}

TEST(BasicAssumptions, ExpressionTemplates6)
{
  UGLY_EXPR_TEST_UNARY_DOUBLE_RETURN(LengthSqr, 2, 2)
  ASSERT_NEAR(result, 8., 1.e-3);
}


TEST(BasicAssumptions, ExpressionTemplates7)
{
  UGLY_EXPR_TEST_UNARY_VEC_RETURN(Normalized, 2, 0)
  ASSERT_NEAR(result[0], 1., 1.e-3);
}


TEST(BasicAssumptions, ExpressionTemplates8)
{
  UGLY_EXPR_TEST_BINARY_VEC_RETURN(Reflected, 1, 1, 1, 0)
  ASSERT_NEAR(result[1], 0., 1.e-3); // Would be -1 but since the return is an expression type, we expect 0.
}


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
  ASSERT_TRUE(d.isFinite().all()); // Only here to silence unused variable warning.
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


TEST(VeryStrongTypedef, WrappingAStructType)
{
  struct TagNumberType {};
  struct TagOtherType {};
  using NumberType = very_strong_typedef<double, TagNumberType>;
  NumberType nt{5.};
  
  struct Test
  {
    double f;
    Test(double _f) : f(_f) {}
  };
  using OtherType = very_strong_typedef<Test, TagOtherType>;
  OtherType ot{5.};
  EXPECT_EQ(value(ot).f, 5.);
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


TEST(Boost, Variant)
{
  boost::variant<int, double> test = 2.;
  boost::get<double>(test) = 3.;
  ASSERT_EQ(test.which(), 1);
  ASSERT_ANY_THROW(int &a = boost::get<int>(test));
  test = 1;
  int &a = boost::get<int>(test);
  a = 5;
  ASSERT_EQ(test.which(), 0);
  std::cout << test << std::endl;
}

namespace VariantTestDetail
{

struct Demo
{
  int &ref;
  int r = {42};
  int s = {7};
  Demo(int &_ref) : ref{_ref} {}
};


class VariantTestVisitorIdentifyString : public boost::static_visitor<bool>
{
public:
  // Compileable version of the operator must be available for each type in the variant.
  template<typename T>
  bool operator()(const T &u) const
  {
    return false;
  }
  
  // http://en.cppreference.com/w/cpp/language/overload_resolution
  // Section "Best viable function" states that normal functions are prefered over templates (?).
  bool operator()(const std::string &u) const
  {
    return true;
  }
};

}

TEST(Boost, VariantVisitation)
{
  using namespace VariantTestDetail;
  int reference_target = 0;
  // See "Enabling Optimizations" section in the manual. Nothrow copy constructors on each type should guarantee best efficiency, i.e. no allocations.
  //static_assert(boost::has_nothrow_constructor<Demo>() == true, "Efficiency. See Manual."); // Triggered because no default ctor
  static_assert(boost::has_nothrow_copy<Demo>() == true, "Efficiency. See Manual.");
  
  using VariantType = boost::variant<Demo, std::string, int>;
  VariantType test_str{"foo"};
  VariantType test_int{1};
  VariantType test_demo{Demo{reference_target}};
  EXPECT_EQ(boost::apply_visitor(VariantTestVisitorIdentifyString(), test_str), true);
  EXPECT_EQ(boost::apply_visitor(VariantTestVisitorIdentifyString(), test_int), false);
  EXPECT_EQ(boost::apply_visitor(VariantTestVisitorIdentifyString(), test_demo), false);
  
  struct VariantAsMember
  {
    VariantType vari = {"bazz"};
  } foo;
}


namespace VariantTestDetail
{
struct PolyBase
{
  virtual int anumber() const = 0;
};

struct PolyDerived1 : public PolyBase
{
  int anumber() const override { return 1; }
};

struct PolyDerived2 : public PolyBase
{
  int anumber() const override { return 2; }
};

struct PolyDerived21 : public PolyDerived2
{
  int anumber() const override { return 21; }
};

struct JustDataBase
{
  int a = 42;
};

struct JustDataDerived : public JustDataBase
{
  int b = 7;
};

};

TEST(Boost, VariantPolymorphic)
{
  using namespace VariantTestDetail;
  using StorageType = boost::variant<PolyBase, PolyDerived1, PolyDerived2, PolyDerived21>;
  StorageType stack_based_storage{PolyDerived2{}};
  EXPECT_EQ(boost::polymorphic_get<PolyBase>(stack_based_storage).anumber(), 2);
  
  using DataStorage = boost::variant<JustDataBase, JustDataDerived>;
  DataStorage data_storage{JustDataDerived{}};
  EXPECT_EQ(boost::polymorphic_get<JustDataBase>(data_storage).a, 42);
  
  data_storage = JustDataBase{};
  EXPECT_EQ(boost::polymorphic_get<JustDataBase>(data_storage).a, 42);
  ASSERT_ANY_THROW(boost::polymorphic_get<JustDataDerived>(data_storage).b); // Boom. Query inactive type.
}


TEST(StrFormat, Test)
{
  EXPECT_EQ(strformat("a%b%c", 1, 2), "a1b2c");
  EXPECT_EQ(strformat("%%%", 1), "%1");
  EXPECT_EQ(strformat("nothing"), "nothing");
  EXPECT_EQ(strformat("%bar%", "foo", "baz"), "foobarbaz");
}

//////////////////////////////////////////////
//// Ptree building in style
//////////////////////////////////////////////
namespace make_pt_detail
{
namespace pt = boost::property_tree;
  
inline void put_pt(pt::ptree &dst)
{
}

template<class T, class ... Args>
inline void put_pt(pt::ptree &dst, const char* name, const T &x, Args&& ... args)
{
  dst.put(name, x);
  put_pt(dst, args...);
}

}

template<class ... Args>
boost::property_tree::ptree make_pt(Args&& ...args)
{
  namespace pt = boost::property_tree;
  pt::ptree dst;
  make_pt_detail::put_pt(dst, args...);
  return dst;
}


TEST(PTree, MakeInStyle)
{
  namespace pt = boost::property_tree;
  pt::ptree tree = make_pt("foo", Spectral3{1.}, "bar", 1., "baz", 42);
  // EXPECT_EQ(tree.get<Spectral3>("foo")[0], 1.);  // Bah! Compile error. Needs operator>> :-(
  EXPECT_EQ(tree.get<double>("bar"), 1.);
  EXPECT_EQ(tree.get<int>("baz"), 42);
}


//////////////////////////////////////////////
////// Demoing Parameterized Tests
//////////////////////////////////////////////
namespace ParameterizedTestDemoDetail
{
using Options = std::unordered_map<const char*, boost::any>;
using Factory = std::function<int*()>;

using Params = std::tuple<double, Factory, Options>;

Params MakeParams(double x, Factory f, std::initializer_list<Options::value_type> opts)
{
  return Params{x, f, Options{std::move(opts)}};
}


class ParameterizedTestDemo : public testing::TestWithParam<ParameterizedTestDemoDetail::Params>
{
};

TEST_P(ParameterizedTestDemo, Foo) 
{
  auto param = GetParam();
  double x;
  ParameterizedTestDemoDetail::Factory f;
  Options opt;
  std::tie(x, f, opt) = param;
  int *i = f();
}

INSTANTIATE_TEST_CASE_P(TheInstiationName,
                        ParameterizedTestDemo,
                        ::testing::Values(
                          MakeParams(1., []{ return new int(42); }, {{"foo", 666}, {"bar", Spectral3{7.}}}), 
                          MakeParams(2., []{ return new int(7); }, {{"foo", 42},{"bar", Spectral3{23.}}})
                        ));
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#pragma once

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
#include <unordered_map>
#include <string_view>
#include <atomic>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <boost/functional/hash.hpp>
//#include <boost/align/aligned_allocator.hpp>


#ifdef __GNUC__
#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x) __builtin_expect(!!(x), 1)
#else
#define unlikely(x) x
#define likely(x) x
#endif

namespace util
{

template<class T>
inline T Sqr(const T &x)
{
  return x*x;
}

template<class T>
inline T Cubed(const T &x)
{
  return x*x*x;
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

inline constexpr double Pow(double x, std::uint32_t e)
{
  // This is the binary exponentiation algorithm
  // https://de.wikipedia.org/wiki/Bin%C3%A4re_Exponentiation
  std::uint32_t bit = 1<< (sizeof(e)*8-1);
  double ret = 1.;
  while (bit)
  {
    ret = ret*ret;
    ret = (e&bit) ? ret*x : ret;
    bit >>= 1;
  }
  return ret;
}


template<class T>
inline constexpr T Modulus(T a, T m, std::enable_if_t<std::is_integral_v<T> && std::is_signed_v<T>>* = nullptr)
{
  //  Example for operator%: -5 % 3 = -2
  //  Should be one though. Can simply add m if result is negative.
  const T tmp = a % m;
  return tmp < 0 ? tmp + m : tmp;
}

template<class T>
inline constexpr T Modulus(T a, T m, std::enable_if_t<std::is_integral_v<T> && std::is_unsigned_v<T>>* = nullptr)
{
  return a % m;
}

template<class T>
inline constexpr T Modulus(T a, T m, std::enable_if_t<std::is_floating_point_v<T>>* = nullptr)
{
  //  Example for operator%: -5 % 3 = -2
  //  Should be one though. Can simply add m if result is negative.
  const T tmp = std::fmod(a, m);
  return tmp < 0 ? tmp + m : tmp;
}


// t = 0: Returns a
// t = 1: Returns b
// otherwise linear inter/extra-polation
template<class T, class U>
inline T Lerp(const T &a, const T &b, U t)
{
  return (std::remove_reference_t<U>(1) - t) * a + t * b;
}


// Note: Will happily take the signbit from zero. So the result for 0 is basically random.
template<class T, typename std::enable_if_t<std::is_floating_point<T>{}, int> = 0>
inline T Sign(const T &x)
{
  return std::copysign(T(1.), x);
}


// Also from PBRT. Used to compute error bounds for floating point arithmetic. See pg. 216.
template<class T>
inline constexpr T Gamma(int n) {
    constexpr T eps_half = std::numeric_limits<T>::epsilon();
    return (n * eps_half) / (1 - n * eps_half);
}


template<class T>
inline bool Quadratic(T a, T b, T c, T &t0, T &t1)
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


inline bool startswith(const std::string &a, const std::string &b)
{
  if (a.size() < b.size())
    return false;
  return a.substr(0, b.size()) == b;
}

inline bool endswith(const std::string &a, const std::string &b)
{
  if (a.size() < b.size())
    return false;
  return a.substr(a.size()-b.size(), b.size()) == b;
}


//  There is no hash support for pairs in the STL.
template<class A, class B>
struct pair_hash
{
  std::size_t operator()(const std::pair<A,B> &v) const
  {
    std::size_t seed = boost::hash_value(v.first);
    boost::hash_combine(seed, boost::hash_value(v.second));
    return seed;
  }
};


// Copy & Paste Fu! https://stackoverflow.com/questions/27140778/range-based-for-with-pairiterator-iterator
template <typename I>
struct iter_pair : std::pair<I, I>
{ 
    using std::pair<I, I>::pair;

    I begin() { return this->first; }
    I end() { return this->second; }
};

#if 1
template<class T, size_t alignment>
class AlignedAllocator
{
  public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    static constexpr size_t ComputeTrueAlignment()
    {
      // By default we may have some random align a. But if type specify 
      // alignas(b) with b>a, I'd get crashes without this little correction.
      if constexpr (std::is_void_v<T>)
        return alignment;
      else
        return std::max(alignment, alignof(T));
    }

    static constexpr size_t true_alignment = ComputeTrueAlignment();
    // Check requirements for posix_memalign.
    static_assert(true_alignment % sizeof(void*) == 0);

    template<class U>
    struct rebind {
      typedef AlignedAllocator<U, alignment> other;
    };

    constexpr AlignedAllocator() noexcept = default;

    template <class U>
    constexpr AlignedAllocator(const AlignedAllocator<U, alignment>&) noexcept
    {
    }

    [[nodiscard]] T* allocate(std::size_t n)
    {
      #ifdef _MSC_VER
            return (T*)_aligned_malloc(n*sizeof(T), true_alignment);
      #else
            void* result = nullptr;
            posix_memalign(&result, true_alignment, n*sizeof(T)); // returns 0 on success. Not using it obviously.
            return (T*)result;
      #endif
    }

    void deallocate(T* p, std::size_t n)
    {
      #ifdef _MSC_VER
        _aligned_free(p);
      #else
        free(p);
      #endif
    }
};

template <class T, class U, size_t a>
bool operator==(const AlignedAllocator<T,a>&, const AlignedAllocator<U,a>&) noexcept
{
  return true;
}

template <class T, class U, size_t a>
bool operator!=(const AlignedAllocator<T, a>&, const AlignedAllocator<U, a>&) noexcept
{
  return false;
}
#else
template<class T, size_t a>
using AlignedAllocator = boost::alignment::aligned_allocator<T,a>;
#endif


// std::vector with 16 byte aligment as required by Eigen's fixed size types.
// This class also comes with range checking in debug mode.
template<class T, class Alloc = AlignedAllocator<T,16>>
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
    auto scope_exit_ ## __LINE__ = util::MakeScopeExit([=](){ code })

    
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


template<class T>
struct enable_if_has_size_member
{
    using type = decltype(std::declval<T&>().size());
};


template<class Container>
inline int isize(const Container &c, typename enable_if_has_size_member<Container>::type = 0)
{
    return static_cast<int>(c.size());
}

template<class Container>
inline long lsize(const Container &c, typename enable_if_has_size_member<Container>::type = 0)
{
  return static_cast<long>(c.size());
}


template<class T>
struct enable_if_has_insert_begin_and_end_members
{
  using type = decltype(std::declval<T&>().insert(
    std::declval<T&>().begin(),
    std::declval<T&>().end(),
    std::declval<T&>().begin() // here this makes actually sense.
  ));
};


template<class Container, typename = typename enable_if_has_insert_begin_and_end_members<Container>::type>
inline void Append(Container &a, const Container &b)
{
  a.insert(a.end(), b.begin(), b.end());
}

template<class T, class Alloc>
inline void PushBackToEnsureSize(std::vector<T, Alloc> &v, std::size_t required_size, const T &filler)
{
  assert(required_size >= v.size());
  std::fill_n(std::back_inserter(v), required_size - v.size(), filler);
}


template<class T>
struct has_begin_end
{
  using type1 = decltype(std::declval<T&>().begin());
  using type2 = decltype(std::declval<T&>().end());
  static constexpr bool value = true;
};


template<class Container, class Func, typename = std::enable_if_t<has_begin_end<Container>::value>>
inline auto TransformVector(Container &a, Func &&f)
{
  using TIn = typename Container::reference;
  using TOut = std::invoke_result_t<Func, TIn>;
  using OutContainer = ToyVector<TOut>;
  OutContainer result;
  std::transform(a.begin(), a.end(), std::back_inserter(result), f);
  return result;
}

// Adapted from https://stackoverflow.com/questions/41660062/how-to-construct-an-stdarray-with-index-sequence
// Use variadic templates and a pack of integers to build the output array without 
// invoking default c'tors.
namespace detail {
  template<typename T, typename U, typename F, std::size_t... Is>
  constexpr auto transform_array(F& f, const U* input, std::index_sequence<Is...>)
     -> std::array<T, sizeof...(Is)> 
  {
    return {{f(input[std::integral_constant<std::size_t, Is>{}])...}};
  }
}


template<class Func, class T, std::size_t n>
inline constexpr auto TransformArray(std::array<T,n> &a, Func &&f)
{
  using TOut = std::invoke_result_t<Func, T>;
  return detail::transform_array<TOut>(f, a.data(), std::make_index_sequence<n>{});
}


// From https://stackoverflow.com/questions/41660062/how-to-construct-an-stdarray-with-index-sequence
// Use variadic templates and a pack of integers to build the output array without 
// invoking default c'tors. 
namespace detail {
  template<typename T, typename F, std::size_t... Is>
  constexpr auto generate_array(F& f, std::index_sequence<Is...>)
   -> std::array<T, sizeof...(Is)> {
    return {{f(std::integral_constant<std::size_t, Is>{})...}};
  }
}

template<std::size_t N, typename F>
inline constexpr auto GenerateArray(F &&f) {
  using TOut = std::invoke_result_t<F, std::size_t>;
  return detail::generate_array<TOut>(f, std::make_index_sequence<N>{});
}


/*
 * Perform an atomic addition to the float via spin-locking
 * on compare_exchange_weak. Memory ordering is release on write
 * consume on read
 *
 * from https://www.reddit.com/r/cpp/comments/338pcj/atomic_addition_of_floats_using_compare_exchange/
 */
inline float AtomicAdd(std::atomic<float> &f, float d) {
  float old = f.load(std::memory_order_consume);
  float desired = old + d;  while (!f.compare_exchange_weak(old, desired,
    std::memory_order_release, std::memory_order_consume))
  {
    desired = old + d;
  }
  return desired;
}


// From https://arne-mertz.de/2018/05/overload-build-a-variant-visitor-on-the-fly/
template <class ...Fs>
struct Overload : Fs... {
  template <class ...Ts>
  Overload(Ts&& ...ts) : Fs{std::forward<Ts>(ts)}...
  {} 

  using Fs::operator()...;
};

template <class ...Ts>
Overload(Ts&&...) -> Overload<std::remove_reference_t<Ts>...>;

// TODO: check if we have an iterator in It
template<class It, class Trafo>
inline std::string Join(const std::string &sep, It begin, It end, Trafo trafo)
{
  if (begin == end)
    return {};
  std::ostringstream os;
  It prev = begin;
  ++begin;
  while (begin != end)
  {
    os << trafo(*prev) << sep;
    prev = begin;
    ++begin;
  }
  os << trafo(*prev);
  return os.str();
}


} // namespace util

using util::Overload;
using util::Sqr;
using util::Cubed;
using util::Heaviside;
using util::Rcp;
using util::Lerp;
using util::Sign;
using util::ToyVector;
using util::isize;
using util::lsize;
using util::AlignedAllocator;
using util::ASSERT_NOT_NULL;
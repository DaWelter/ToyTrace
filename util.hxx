#pragma once

#include <iostream>
#include <cassert>
#include <malloc.h>

template<class T>
inline T Sqr(const T &x)
{
  return x*x;
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

// Alternatively I could use 
// boost::alignment::aligned_allocator.
template <typename T, std::size_t alignment>
class AlignmentAllocator : public std::allocator<T>
{
    public:
    typedef typename std::allocator<T>::pointer pointer;
    typedef typename std::allocator<T>::size_type size_type;
    typedef typename std::allocator<T>::value_type value_type;

    template<typename _Tp1>
    struct rebind
    {
        typedef AlignmentAllocator<_Tp1, alignment> other;
    };

    pointer allocate(size_type n, const void *hint=0)
    {
      return (pointer)memalign(alignment, n*sizeof(value_type));
    }

    void deallocate(pointer p, size_type n)
    {
      free(p);
    }

//     AlignmentAllocator() throw() : std::allocator<T>()
//     {
//     }
// 
//     AlignmentAllocator(const AlignmentAllocator &a) throw() : std::allocator<T>(a)
//     { }
// 
//     template <class U>
//     AlignmentAllocator(const AlignmentAllocator<U> &a) throw() : std::allocator<T>(a)
//     { }
};


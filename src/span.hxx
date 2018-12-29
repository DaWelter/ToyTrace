#pragma once

#include "util.hxx"

// Inspired by Microsofts GSL. I don't use that one because I read that
// it has runtime checks that are hard to disable, if at all. My version
// is as lightweight as possible and very very simple.
template<class T>
class Span
{
  using size_t = std::size_t;
  
  T* _begin;
  size_t _size;

public:
  Span(T* begin, size_t size)
    : _begin{begin}, _size{size}
  {}
  
  T operator[](const int idx) const 
  {
    assert (idx >= 0 && (size_t)idx < _size);
    return _begin[idx];
  }
  
  T& operator[](const int idx)
  {
    assert (idx >= 0 && (size_t)idx < _size);
    return _begin[idx];
  }
  
  size_t size() const
  {
    return _size;
  }
  
  T* begin() { return _begin; }
  
  const T* begin() const { return _begin; }
  
  operator Span<const T>() { return Span<const T>(begin(), size()); }
};

template<class T, class Alloc>
inline Span<const T> AsSpan(const std::vector<T, Alloc> &v)
{
  return Span<const T>(v.size()>0 ? &v[0] : nullptr, v.size());
}

template<class T, class Alloc>
inline Span<T> AsSpan(std::vector<T, Alloc> &v)
{
  return Span<T>(v.size()>0 ? &v[0] : nullptr, v.size());
}

template<class T>
inline Span<T> Subspan(Span<T> s, int offset, int size)
{
  assert((offset >= 0) && (offset + size <= s.size()));
  return Span<T>(s.begin() + offset, size);
}

template<class T>
inline bool IsInRange(Span<T> s, T* p)
{
  auto offset = p - s.begin();
  return offset>=0 && offset < s.size();
}

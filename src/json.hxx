#pragma once

#include "vec3f.hxx"

#include <string_view>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/prettywriter.h>


namespace rapidjson_util
{
namespace rj = rapidjson;

using Alloc = rj::Document::AllocatorType; // For some reason I must supply an "allocator" almost everywhere. Why? Who knows?!

inline std::string ToString(rj::Document &doc)
{
  rj::StringBuffer buffer;
  rj::PrettyWriter<rj::StringBuffer> writer(buffer);
  doc.Accept(writer);
  return { buffer.GetString(), buffer.GetSize() };
}

template<class Derived>
inline rapidjson::Value ToJSON(const Eigen::ArrayBase<Derived> &v, Alloc &alloc)
{
  rj::Value json_vec(rj::kArrayType);
  if(v.cols() == 1)
  {
    for (Eigen::Index i=0; i<v.rows(); ++i) {
          json_vec.PushBack(rj::Value(v(i)).Move(), alloc);
    }
  }
  else
  {
    for (Eigen::Index i=0; i<v.rows(); ++i) 
    {
      rj::Value json_col(rj::kArrayType);
      for (Eigen::Index j=0; j<v.cols(); ++j) 
      {
        json_col.PushBack(v(i,j), alloc);
      }
      json_vec.PushBack(json_col, alloc);
    }
  }
  return json_vec;
}

template<class T, int rows, int cols>
inline rapidjson::Value ToJSON(const Eigen::Matrix<T, rows, cols> &v, Alloc &alloc)
{
    return ToJSON(v.array(), alloc);
}


template<class T>
inline rapidjson::Value ToJSON(const T &v, Alloc &alloc, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr)
{
  return rj::Value(v);
}


inline rapidjson::Value ToJSON(std::string s, Alloc &alloc)
{
    rj::Value js; 
    js.SetString(s.c_str(), rj::SizeType(s.size()), alloc);
    return js;
}

inline rapidjson::Value ToJSON(std::string_view s, Alloc &alloc)
{
  return ToJSON(std::string(s), alloc);
}


// From https://stackoverflow.com/questions/38077228/how-can-i-know-the-c-template-is-a-container-or-a-type
template<typename ...>
using to_void = void; // maps everything to void, used in non-evaluated contexts

template<typename T, typename = void>
struct is_container : std::false_type
{};

template<typename T>
struct is_container<T,
        to_void<decltype(std::declval<T>().begin()),
                decltype(std::declval<T>().end()),
                typename T::value_type
        >> : std::true_type // will  be enabled for iterable objects
{};


template<class Container>
inline rj::Value ToJSON(const Container &c, Alloc &alloc, typename std::enable_if<is_container<Container>::value>::type* = nullptr)
{
  rj::Value json_container(rj::kArrayType);
  for (const auto &x : c)
  {
    json_container.PushBack(ToJSON(x, alloc), alloc);
  }
  return json_container;
}

} // namespace rapidjson_util
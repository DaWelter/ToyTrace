#pragma once

#include "types.hxx"

#include <boost/functional/hash.hpp>

struct Material
{
  Shader* shader = {nullptr};
  Medium* medium = {nullptr};
  RadianceOrImportance::AreaEmitter *emitter = {nullptr};
  
  struct Hash
  {
    inline std::size_t operator()(const Material &key) const
    {
      std::size_t h = boost::hash_value((key.shader));
      boost::hash_combine(h, boost::hash_value((key.medium)));
      boost::hash_combine(h, boost::hash_value((void*)key.emitter));
      return h;
    }
  };
  
  friend inline bool operator==(const Material &a, const Material &b)
  {
    return (a.shader == b.shader) &&
          (a.medium == b.medium) &&
          (a.emitter == b.emitter);
  }
};

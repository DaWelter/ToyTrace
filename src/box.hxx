#ifndef BOX_HXX
#define BOX_HXX

#include "vec3f.hxx"
#include <limits>

class Box
{
public:
  Double3 min,max;
  
  Box() 
  { Clear(); };
  
  inline Double3 Center() const
  {
    return 0.5*(min+max);
  }
  
  void Extend(Double3 a)
  { min = a.cwiseMin(min); max = a.cwiseMax(max); };
  
  void Clear()
  { 
    constexpr auto infty = std::numeric_limits<Double3::value_type>::infinity();
    min = Double3( infty, infty, infty); 
    max = Double3(-infty,-infty,-infty); 
  };
  
  void Extend(Box box)
  {
    Extend(box.min);
    Extend(box.max);
  };

  bool InBox(const Box &box) const
  {
	  return (box.max[0] > max[0] &&
            box.max[1] > max[1] && 
            box.max[2] > max[2] && 
            box.min[0] < min[0] &&
            box.min[1] < min[1] &&
            box.min[2] < min[2]);

  }

  bool InBox(const Double3 &p) const
  {
    return (p.array() > min.array() &&
            p.array() < max.array()).all();
  }

  bool Intersect(const Box &box) const
  {
    // If we don't find a separating axis ...
	  return !( box.min[0] > max[0] ||
              box.min[1] > max[1] ||
              box.min[2] > max[2] ||
              box.max[0] < min[0] ||
              box.max[1] < min[1] ||
              box.max[2] < min[2]) ;
  }
};

#endif

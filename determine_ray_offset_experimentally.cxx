#include "ray.hxx"
#include "image.hxx"
#include "camera.hxx"
#include "infiniteplane.hxx"
#include "sampler.hxx"
#include "scene.hxx"
#include "renderingalgorithms.hxx"
#include "atmosphere.hxx"
#include "util.hxx"


inline RaySegment MakeSegmentAt(const RaySurfaceIntersection &intersection, const Double3 &ray_dir, float eps_factor)
{
  float val = intersection.pos.cwiseAbs().maxCoeff();
  float eps = eps_factor*(val - std::nextafter(val, 0.f));
  assert(eps > 0.f);
  return RaySegment{
    {intersection.pos + eps*AlignedNormal(intersection.geometry_normal, ray_dir), ray_dir},
    LargeNumber
  };
}


bool RunTest(const EmbreeAccelerator &world, Sampler &sampler, float eps_factor, Double3 sphere_org, double sphere_rad)
{
  const int N = 1000;
  for (int num = 0; num < N; ++num)
  {
    // Sample random position outside the sphere as start point.
    // And position inside the sphere as end point.
    // Intersection is thus guaranteed.
    Double3 org = 
      SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare())
      * sphere_rad * 10. + sphere_org;
    Double3 target = 
      SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare())
      * sphere_rad * 0.99 + sphere_org;
    Double3 dir = target - org; 
    Normalize(dir);
    
    // Shoot ray to the inside. Expect intersection at the
    // front side of the sphere.
    RaySegment rs{{org, dir}, LargeNumber};
    RaySurfaceIntersection intersect1;
    bool bhit = world.FirstIntersection(rs.ray, 0, rs.length, intersect1);
    if (!bhit) return false;
    
    // Put the origin at the intersection and shoot a ray to the outside
    // in a random direction. Expect no further hit.
    auto m  = OrthogonalSystemZAligned(intersect1.geometry_normal);
    auto new_dir = m * SampleTrafo::ToUniformHemisphere(sampler.UniformUnitSquare());
    rs = MakeSegmentAt(intersect1, new_dir, eps_factor);
    bhit = world.FirstIntersection(rs.ray, 0, rs.length, intersect1);
    if (bhit) return false;
  }
  return true;
}


int main(int argc, char **argv)
{
  float radius = 1.e-3;
  float radius_increment_factor = 2.;
  float eps_factor_init = 1.;
  Sampler sampler;
  
  while (std::isfinite(radius))
  {

    Double3 sphere_org{1.e3, 0., 0.};
    double sphere_rad = radius;
    Spheres geom;
    geom.Append(sphere_org.cast<float>(), (float)sphere_rad, MaterialIndex{-1});
    EmbreeAccelerator world;
    world.Add(geom);
    world.Build();
    
    bool ok = false;
    float eps_factor = eps_factor_init;
    while (eps_factor < 10000.)
    {
      ok = RunTest(world, sampler, eps_factor, sphere_org, sphere_rad);
      if (ok)
        break;
      eps_factor *= 2.;
    }
    std::cout << sphere_rad << " " << (ok ? eps_factor : std::numeric_limits<float>::quiet_NaN()) << std::endl;
    
    radius *= radius_increment_factor;
  }
}
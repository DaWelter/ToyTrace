#include "primitive.hxx"
#include "sampler.hxx"
#include "ray.hxx"
#include "scene.hxx"

// Mesh::Mesh()
//   : Geometry{Geometry::PRIMITIVES_TRIANGLES}
// {}


Mesh::Mesh(index_t num_triangles, index_t num_vertices)
  : Geometry{Geometry::PRIMITIVES_TRIANGLES}
{
  // Using Eigen::NoChange because number of columns is fixed by compile time constant.
  vertices.resize(num_vertices, Eigen::NoChange);
  vert_indices.resize(num_triangles, Eigen::NoChange);
  normals.resize(num_vertices, Eigen::NoChange);
  uvs.resize(num_vertices, Eigen::NoChange);
}


// Mesh::Mesh(Mesh &&other) 
//   : Geometry{other},
//     vertices{std::move(other.vertices)},
//     vert_indices{std::move(other.vert_indices)},
//     normals{std::move(other.normals)},
//     uvs{std::move(other.uvs)}
//     //material_indices{std::move(other.material_indices)}
// {
// }
// 
// 
// Mesh& Mesh::operator=(Mesh &&other)
// {
//   if (&other == this)
//     return *this;
//   this->Geometry::operator=(other);
//   vertices = std::move(other.vertices);
//   vert_indices = std::move(other.vert_indices);
//   normals = std::move(other.normals);
//   uvs = std::move(other.uvs);
//   //material_indices = std::move(material_indices);
//   return *this;
// }


void Mesh::MakeFlatNormals()
{
  auto old_vertices = vertices;
  auto old_uvs      = uvs;
  auto num_vertices = NumTriangles()*3;
  vertices.resize(num_vertices, Eigen::NoChange);
  normals.resize(num_vertices, Eigen::NoChange);
  uvs.resize(num_vertices, Eigen::NoChange);
  index_t k = 0;
  for (index_t i=0; i<NumTriangles(); ++i)
  {
    auto a = vert_indices(i, 0);
    auto b = vert_indices(i, 1);
    auto c = vert_indices(i, 2);
    Float3 d1 = old_vertices.row(b)-old_vertices.row(a);
    Float3 d2 = old_vertices.row(c)-old_vertices.row(a);
    Float3 n = Normalized(Cross(d1, d2));
    for (index_t j=0; j<3; ++j)
    {
      auto v_idx = vert_indices(i, j);
      vertices.row(k) = old_vertices.row(v_idx);
      normals.row(k) = n;
      uvs.row(k) = old_uvs.row(v_idx);
      vert_indices(i, j) = k;
      ++k;
    }
  }
}


void Mesh::Append(const Geometry & other)
{
  if (other.type != this->type)
    throw std::runtime_error("Geometry type mismatch!");
  this->Append(static_cast<const Mesh&>(other));
}


template<class M>
static void AppendVertical(M &a, const M &b)
{
  // Did NOT work for me: https://stackoverflow.com/questions/21496157/eigen-how-to-concatenate-matrix-along-a-specific-dimension
  auto prev_a_rows = a.rows();
  M c(a.rows()+b.rows(), a.cols());
  c << a, b;
  a.resizeLike(c); // Resizing uninitializes the matrix! Unlike e.g. numpy where the data is copied over.
  a = c;
}
  

void  Mesh::Append(const Mesh &other)
{
  //if (other.material_index != material_index)
  //  throw std::runtime_error("Cannot merge geometries with mismatching materials");
  // a + b > overflow => b > overflow - a
  if (other.NumTriangles() > std::numeric_limits<decltype(NumTriangles())>::max() - NumTriangles())
    throw std::range_error("Cannot handle that many triangles in a mesh.");
  if (other.NumVertices() > std::numeric_limits<decltype(NumVertices())>::max() - NumVertices())
    throw std::range_error("Cannot handle that many vertices in a mesh.");
  auto tri_start = vert_indices.rows();
  auto vert_start = vertices.rows();
  AppendVertical(vertices, other.vertices);
  AppendVertical(normals, other.normals);
  AppendVertical(uvs, other.uvs);
  AppendVertical(vert_indices, other.vert_indices);
  for (auto i=tri_start; i<vert_indices.rows(); ++i)
  {
    vert_indices(i, 0) += vert_start;
    vert_indices(i, 1) += vert_start;
    vert_indices(i, 2) += vert_start;
  }
}


void Mesh::GetLocalGeometry(SurfaceInteraction& ia) const
{
  assert(ia.hitid.geom == this);
  assert(ia.hitid.index >= 0 && ia.hitid.index < Size());
  const auto a = vert_indices(ia.hitid.index, 0),
             b = vert_indices(ia.hitid.index, 1),
             c = vert_indices(ia.hitid.index, 2);
  const Float3 f = ia.hitid.barry.cast<float>();
  const Float3 pos =  f[0] * vertices.row(a) +
                      f[1] * vertices.row(b) +
                      f[2] * vertices.row(c);
  ia.pos = pos.cast<double>();
  const Float3 normal = Normalized(Cross(vertices.row(b)-vertices.row(a),vertices.row(c)-vertices.row(a)));
  ia.geometry_normal = normal.cast<double>();
  const Float3 smooth_normal = Normalized(
                    normals.row(a)*f[0]+
                    normals.row(b)*f[1]+
                    normals.row(c)*f[2]);
  ia.smooth_normal = smooth_normal.cast<double>();
  assert  (ia.pos.allFinite());
  assert  (ia.geometry_normal.allFinite());
  assert  (ia.smooth_normal.allFinite());
  ASSERT_NORMALIZED(ia.geometry_normal);
  ASSERT_NORMALIZED(ia.smooth_normal);
  FillPosBoundsTriangle(ia, vertices.row(a), vertices.row(b), vertices.row(c));
}


HitId Mesh::SampleUniformPosition(index_t index, Sampler &sampler) const
{
  return HitId{
    this,
    index,
    SampleTrafo::ToTriangleBarycentricCoords(sampler.UniformUnitSquare())
  };
}


double Mesh::Area(index_t index) const
{
  assert(index >= 0 && index < Size());
  const auto a = vert_indices(index, 0),
             b = vert_indices(index, 1),
             c = vert_indices(index, 2);
  const Float3 n(Cross(vertices.row(b)-vertices.row(a),vertices.row(c)-vertices.row(a)));
  return 0.5*Length(n);
}


std::unique_ptr<Geometry> Mesh::Clone() const
{
  return std::make_unique<Mesh>(*this);
}


void AppendSingleTriangle(
  Mesh &dst, const Float3 &a, const Float3 &b, const Float3 &c, const Float3 &n)
{
  Mesh m{1, 3};
  m.vertices.transpose() <<  a, b, c;
  m.uvs.transpose() << Float2{a[0], a[1]}, Float2{b[0], b[1]}, Float2{c[0], c[1]};
  m.normals.transpose() << n, n, n;
  m.vert_indices.transpose() << UInt3{0,1,2};
  dst.Append(m);
}


//////////////////////////////////////////////////////////////////////

Spheres::Spheres()
  : Geometry(Geometry::PRIMITIVES_SPHERES)
{
  
}

void Spheres::Append(const Float3 pos, const float radius)
{
  if (Size() >= std::numeric_limits<decltype(Size())>::max())
    throw std::range_error("Cannot handle that many spheres in a geometry.");
  spheres.push_back(Vector4f{ pos[0], pos[1], pos[2], radius });
}


void Spheres::Append(const Spheres& other)
{
  //if (other.material_index != material_index)
  //  throw std::runtime_error("Cannot merge geometries with mismatching materials");
  if (other.Size() > std::numeric_limits<decltype(Size())>::max() - Size())
    throw std::range_error("Cannot handle that many spheres in a geometry.");
  spheres.insert(spheres.end(), other.spheres.begin(), other.spheres.end());
}


void Spheres::Append(const Geometry & other)
{
  if (other.type != this->type)
    throw std::runtime_error("Geometry type mismatch!");
  this->Append(static_cast<const Spheres&>(other));
}


HitId Spheres::SampleUniformPosition(index_t index, Sampler &sampler) const
{
  assert(index >= 0 && index < spheres.size());
  float rad     = spheres[index][3];
  Double3 barry = rad*SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
  return HitId{
    this,
    index,
    barry.cast<double>()
  };
}


double Spheres::Area(index_t index) const
{
  assert(index >= 0 && index < spheres.size());
  float radius = spheres[index][3];
  return Sqr(radius)*UnitSphereSurfaceArea;
}


void Spheres::GetLocalGeometry(SurfaceInteraction& ia) const
{
  assert(ia.hitid.geom == this);
  Float3 center = spheres[ia.hitid.index].head<3>(); 
  float rad     = spheres[ia.hitid.index][3];  
  ia.geometry_normal = Normalized(ia.hitid.barry);
  ia.smooth_normal = ia.geometry_normal;
  ia.pos = (center.cast<double>() + ia.hitid.barry);
  FillPosBoundsSphere(ia);
}

std::unique_ptr<Geometry> Spheres::Clone() const
{
  return std::make_unique<Spheres>(*this);
}


//////////////////////////////////////////////////////////////////////
void FillPosBoundsTriangle(SurfaceInteraction &interaction, const Float3 &p0, const Float3 &p1, const Float3 &p2)
{
  // Compute the position error bounds, according to PBRT Chpt 3.9, pg. 227
  auto b = interaction.hitid.barry.cast<float>();
  interaction.pos_bounds =
    ((p0*b[0]).cwiseAbs() +
     (p1*b[1]).cwiseAbs() +
     (p2*b[2]).cwiseAbs())*Gamma<float>(7);
}


void FillPosBoundsSphere(SurfaceInteraction &interaction)
{
  // PBRT. pg.225 Chpt. 3
  interaction.pos_bounds = interaction.pos.cast<float>().cwiseAbs()*Gamma<float>(5); 
}


//////////////////////////////////////////////////////////////////////
std::tuple<bool, double, double> ClipRayToSphereInterior(const Double3 &ray_org, const Double3 &ray_dir, double tnear, double tfar, const Double3 &sphere_p, double sphere_r)
{
  Double3 v = ray_org - sphere_p;
  ASSERT_NORMALIZED(ray_dir);
  const double A = 1.; //Dot(ray_dir,ray_dir);
  const double B = 2.0*Dot(v,ray_dir);
  const double C = Dot(v,v) - Sqr(sphere_r);
  //const float Berr = DotAbs(v, ray_dir)*Gamma<float>(3);
  //const float Cerr = (2.f*Dot(v,v) + Sqr(sphere_r))*Gamma<float>(3);
  double t0, t1; //, err0, err1;
  const bool has_solution = Quadratic(A, B, C, t0, t1);
  if (!has_solution || tfar <= t0 || tnear >= t1)
  {
    return std::make_tuple(false, tnear, tfar);
  }
  else
  {
    if (tnear < t0)
    {
      tnear = t0;
    }
    if (tfar > t1)
    {
      tfar = t1;
    }
    assert(tfar > tnear);
    return std::make_tuple(true, tnear, tfar);
  }
}


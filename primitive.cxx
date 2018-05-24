#include "primitive.hxx"
#include "sampler.hxx"
#include "ray.hxx"

// Mesh::Mesh()
//   : Geometry{Geometry::PRIMITIVES_TRIANGLES}
// {}


Mesh::Mesh(int num_triangles, int num_vertices)
  : Geometry{Geometry::PRIMITIVES_TRIANGLES}
{
  // Using Eigen::NoChange because number of columns is fixed by compile time constant.
  vertices.resize(num_vertices, Eigen::NoChange);
  vert_indices.resize(num_triangles, Eigen::NoChange);
  normals.resize(num_vertices, Eigen::NoChange);
  uvs.resize(num_vertices, Eigen::NoChange);
  material_indices.resize(num_triangles, MaterialIndex{-1});
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
  int num_vertices = NumTriangles()*3;
  vertices.resize(num_vertices, Eigen::NoChange);
  normals.resize(num_vertices, Eigen::NoChange);
  uvs.resize(num_vertices, Eigen::NoChange);
  int k = 0;
  for (int i=0; i<NumTriangles(); ++i)
  {
    auto a = vert_indices(i, 0);
    auto b = vert_indices(i, 1);
    auto c = vert_indices(i, 2);
    Float3 d1 = old_vertices.row(b)-old_vertices.row(a);
    Float3 d2 = old_vertices.row(c)-old_vertices.row(a);
    Float3 n = Normalized(Cross(d1, d2));
    for (int j=0; j<3; ++j)
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
  int tri_start = vert_indices.rows();
  int vert_start = vertices.rows();
  AppendVertical(vertices, other.vertices);
  AppendVertical(normals, other.normals);
  AppendVertical(uvs, other.uvs);
  AppendVertical(vert_indices, other.vert_indices);
  for (int i=tri_start; i<vert_indices.rows(); ++i)
  {
    vert_indices(i, 0) += vert_start;
    vert_indices(i, 1) += vert_start;
    vert_indices(i, 2) += vert_start;
  }
  material_indices.reserve(material_indices.size() + other.material_indices.size());
  for (auto i : other.material_indices)
    material_indices.push_back(i);
}


void Mesh::GetLocalGeometry(SurfaceInteraction& ia) const
{
  assert(ia.hitid.geom == this);
  assert(ia.hitid.index >= 0 && ia.hitid.index < Size());
  const int a = vert_indices(ia.hitid.index, 0),
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
}


HitId Mesh::SampleUniformPosition(int index, Sampler &sampler) const
{
  return HitId{
    this,
    index,
    SampleTrafo::ToTriangleBarycentricCoords(sampler.UniformUnitSquare())
  };
}


double Mesh::Area(int index) const
{
  assert(index >= 0 && index < Size());
  const int a = vert_indices(index, 0),
            b = vert_indices(index, 1),
            c = vert_indices(index, 2);
  const Float3 n(Cross(vertices.row(b)-vertices.row(a),vertices.row(c)-vertices.row(a)));
  return 0.5*Length(n);
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

void Spheres::Append(const Float3 pos, const float radius, MaterialIndex material_index)
{
  spheres.push_back(Vector4f{pos[0], pos[1], pos[2], radius});
  material_indices.push_back(material_index);
}


void Spheres::Append(const Spheres& other)
{
  spheres.insert(spheres.end(), other.spheres.begin(), other.spheres.end());
  material_indices.insert(material_indices.end(), other.material_indices.begin(), other.material_indices.end());
}


HitId Spheres::SampleUniformPosition(int index, Sampler &sampler) const
{
  float rad     = spheres[index][3];
  Double3 barry = rad*SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
  return HitId{
    this,
    index,
    barry.cast<double>()
  };
}


double Spheres::Area(int index) const
{
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
}

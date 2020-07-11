#include "parse_common.hxx"

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>


namespace scenereader
{

void AddDefaultMaterials(Scope &scope, const Scene &scene)
{
  scope.mediums.set_and_activate("default", nullptr);
  scope.shaders.set_and_activate("black", scene.black_shader);
  scope.shaders.set_and_activate("invisible", scene.invisible_shader);
  scope.shaders.set_and_activate("none", nullptr);
  scope.shaders.set_and_activate("default", scene.default_shader);
  scope.areaemitters.set_and_activate("none", nullptr);
  scope.materials["default"] = Material{ 
    /*shader =*/ scene.default_shader };
}


GlobalContext::GlobalContext(Scene &scene_, RenderingParameters *params_, const fs::path &path_hint)
  : scene{ &scene_ }, params{ params_ }, filename{ path_hint }
{
  for (int i = 0; i < scene->GetNumMaterials(); ++i)
    to_material_index[scene->GetMaterial(i)] = MaterialIndex(i);

  if (!filename.empty())
  {
    this->search_paths.emplace_back(
      filename.parent_path());
  }
  else
  {
    this->search_paths.emplace_back(fs::path(""));
  }
  if (params)
  {
    for (auto s : params->search_paths)
      this->search_paths.emplace_back(s);
  }
}


fs::path GlobalContext::MakeFullPath(const fs::path &filename) const
{
  if (filename.is_relative())
  {
    for (auto parent_path : this->search_paths)
    {
      auto trial = parent_path / filename;
      if (fs::exists(trial))
        return trial;
    }
    throw std::runtime_error("Cannot find a file in the search paths matching the name " + filename.string());
  }
  else
    return filename;
}

} // scenereader



namespace scenereader::assimp
{


inline aiMatrix3x3 NormalTrafo(const aiMatrix4x4 &trafo)
{
  aiMatrix3x3 m{ trafo };
  return m.Inverse().Transpose();
}

inline static Double3 aiVector3_to_myvector(const aiVector3D &v)
{
  return Double3{ v[0], v[1], v[2] };
}

bool CheckFace(const aiMesh* aimesh, const aiFace* face)
{
  if (face->mNumIndices != 3)
    throw std::runtime_error("Face has too many vertices. Assimp should have converted to triangle, though ...");
  unsigned vidx[3] = {
    face->mIndices[0],
    face->mIndices[1],
    face->mIndices[2]
  };
  Double3 verts[3];
  for (int i = 0; i < 3; ++i)
  {
    if (vidx[i] >= aimesh->mNumVertices)
      throw std::runtime_error("Invalid face. Vertex index beyond bounds.");
    verts[i] = aiVector3_to_myvector(aimesh->mVertices[vidx[i]]);
  }
  double area = Length(Cross(verts[1] - verts[0], verts[2] - verts[0]));
  return area > 0.;
}


Mesh ReadMesh(const Transform model_transform, const aiMatrix4x4 &ai_mesh_transform, const aiMesh* aimesh)
{
  //Material material = GetMaterial(scope, aimesh);

  //auto m = ndref.local_to_world;
  bool hasuv = aimesh->GetNumUVChannels() > 0;
  bool hasnormals = aimesh->HasNormals();

  auto normal_trafo = scenereader::NormalTrafo(model_transform);
  auto ai_normal_trafo = NormalTrafo(ai_mesh_transform);
  // Note:  (M N)^-1^T = (N^-1 M^-1)^T = M^-1^T N^-1^T (???)

  std::vector<UInt3> vert_indices; vert_indices.reserve(1024);
  for (unsigned int face_idx = 0; face_idx < aimesh->mNumFaces; ++face_idx)
  {
    const aiFace* face = &aimesh->mFaces[face_idx];
    if (CheckFace(aimesh, face))
    {
      vert_indices.push_back(
        UInt3(face->mIndices[0],
          face->mIndices[1],
          face->mIndices[2]));
    }
  }

  Mesh mesh(static_cast<Mesh::index_t>(vert_indices.size()),
    static_cast<Mesh::index_t>(aimesh->mNumVertices));
  for (int i = 0; i < vert_indices.size(); ++i)
    mesh.vert_indices.row(i) = vert_indices[i];

  for (unsigned i = 0; i < aimesh->mNumVertices; ++i)
  {
    Double3 v =
      model_transform * aiVector3_to_myvector(ai_mesh_transform*aimesh->mVertices[i]);
    assert(v.allFinite());
    mesh.vertices.row(i) = v.cast<float>();
  }

  if (hasnormals)
  {
    for (unsigned i = 0; i < aimesh->mNumVertices; ++i)
    {
      Double3 n =
        Normalized(normal_trafo * aiVector3_to_myvector(ai_normal_trafo * aimesh->mNormals[i]));
      assert(n.allFinite());
      mesh.normals.row(i) = n.cast<float>();
    }
  }
  else
    mesh.MakeFlatNormals();

  if (hasuv)
  {
    for (unsigned i = 0; i < aimesh->mNumVertices; ++i)
    {
      Double3 uv =
        aiVector3_to_myvector(
          aimesh->mTextureCoords[0][i]);
      assert(uv.allFinite());
      mesh.uvs(i, 0) = uv[0];
      mesh.uvs(i, 1) = uv[1];
    }
  }
  else
    mesh.uvs.setConstant(0.);
  return mesh;
}


std::optional<std::string> GetMaterialName(const aiScene *aiscene, const aiMesh* mesh)
{
  if (mesh->mMaterialIndex >= aiscene->mNumMaterials)
    return {};
  const aiMaterial* mat = aiscene->mMaterials[mesh->mMaterialIndex];
  aiString ainame;
  mat->Get(AI_MATKEY_NAME, ainame);
  auto name = std::string(ainame.C_Str());
  return name;
}


struct NodeRef
{
  NodeRef(aiNode* _node, const aiMatrix4x4 &_mTrafoParent)
    : node(_node), local_to_world(_mTrafoParent * _node->mTransformation)
  {}
  aiNode* node;
  aiMatrix4x4 local_to_world;
};



void ReadNode(Scene &scene, Transform model_transform, MaterialGetter material_getter, bool material_assignment_by_object_names, const aiScene* aiscene, const NodeRef &ndref)
{
  const auto *nd = ndref.node;
  for (unsigned int mesh_idx = 0; mesh_idx < nd->mNumMeshes; ++mesh_idx)
  {
    const aiMesh* aimesh = aiscene->mMeshes[nd->mMeshes[mesh_idx]];
    fmt::print("Mesh {} ({}), mat_idx={}\n", mesh_idx, aimesh->mName.C_Str(), aimesh->mMaterialIndex);
    auto mesh = ReadMesh(model_transform, ndref.local_to_world, aimesh);
    auto material_name = [&]() -> std::optional<std::string> {
      if (material_assignment_by_object_names)
      {
        return { std::string{aimesh->mName.C_Str()} };
      }
      else
        return GetMaterialName(aiscene, aimesh);
    }();
    scene.Append(mesh, material_getter(material_name));
  }
}


void Read(Scene & scene, Transform model_transform, MaterialGetter material_getter, bool material_assignment_by_object_names, const fs::path & filename_path)
{
  // Example see: https://github.com/assimp/assimp/blob/master/samples/SimpleOpenGL/Sample_SimpleOpenGL.c
  const std::string filename_str = filename_path.string();
  const char* filename = filename_str.c_str();

  std::printf("Reading Mesh: %s\n", filename);
  auto* aiscene = aiImportFile(filename,
    aiProcess_Triangulate
  );

  if (!aiscene)
  {
    throw std::runtime_error(fmt::format("Error: could not load file {}. because: {}", filename, aiGetErrorString()));
  }

  std::vector<NodeRef> nodestack{ { aiscene->mRootNode, aiMatrix4x4{} } };
  while (!nodestack.empty())
  {
    auto ndref = nodestack.back();
    nodestack.pop_back();
    for (unsigned int i = 0; i < ndref.node->mNumChildren; ++i)
    {
      nodestack.push_back({ ndref.node->mChildren[i], ndref.local_to_world });
    }

    ReadNode(scene, model_transform, material_getter, material_assignment_by_object_names, aiscene, ndref);
  }

  aiReleaseImport(aiscene);
}

} //namespace scenereader::assimp
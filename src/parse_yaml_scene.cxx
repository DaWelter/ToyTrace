#include "parse_common.hxx"

namespace scenereader
{

using namespace RadianceOrImportance;
using util::startswith;
using util::endswith;
using std::string;

template<class U>
struct DispatchOverload
{};


struct MaterialMap
{
  Material default_material;
  struct Target
  {
    string name;
    Material mat;
  };
  std::unordered_map<string, Target> name_to_mat;
};


class YamlSceneReader
{
  GlobalContext ctx;
  YAML::Node doc;
public:
  YamlSceneReader(
    Scene& scene,
    RenderingParameters *render_params,
    const fs::path &path_hint)
    : ctx{ scene, render_params, path_hint }
  {
  }

  void Parse(std::istream &is, Scope &scope)
  {
    auto doc = YAML::Load(is);
    ParseScope(doc, scope);
  }

  //using ItemParsFunc = std::function<void (YamlSceneReader*, const YAML::Node &, Scope&)>;
  using ItemParsFunc = void (YamlSceneReader::*)(const YAML::Node &, Scope&);

  void ParseItems(
    const YAML::Node &parent, 
    const char* key, 
    Scope &scope, 
    ItemParsFunc itemparser
  )
  {
    auto listnode = parent[key];
    for (auto it = listnode.begin(); it != listnode.end(); ++it)
    {
      std::invoke(itemparser, this, *it, scope);
      //(this->*itemparser)(*it, scope);
      //itemparser(this, *it, scope);
    }
  }

  std::pair<string, Shader*> ParseItem(const YAML::Node &original_node, const Scope &scope, DispatchOverload<Shader>);
  std::pair<string, materials::Medium*> ParseItem(const YAML::Node &node, const Scope &scope, DispatchOverload<Medium>);
  std::pair<string, AreaEmitter*> ParseItem(const YAML::Node &node, const Scope &scope, DispatchOverload<AreaEmitter>);
  void ParseAndInsertMaterial(const YAML::Node &node, Scope &scope);
  void ParseAndInsertLight(const YAML::Node &node, Scope &scope);
  void ParseAndInsertTransform(const YAML::Node &node, Scope &scope);
  void ParseAndInsertModel(const YAML::Node &original_node, Scope &scope);
  void ParseView(const YAML::Node &node);

  template<class U>
  U* ParseReferenceOrInlined(const YAML::Node node, const string &key, const Scope &scope)
  {
    if (node.IsScalar())
    {
      return Lookup<U>(node, scope, node.as<string>());
    }
    else
    {
      auto[name, shader] = ParseItem(node, scope, DispatchOverload<U>{});
      if (!name.empty())
        throw MakeException(node, "Item must be unnamed");
      return shader;
    }
  }

  template<class U>
  void ParseAndInsertItem(const YAML::Node &original_node, Scope &scope)
  {
    auto[name, item] = ParseItem(original_node, scope, DispatchOverload<U>{});
    if (name.empty())
      throw MakeException(original_node, "Item must be named");
    scope.InsertAndActivate(name, item);
  }

  void ParseScope(const YAML::Node &node, Scope &parent_scope)
  {
    Scope scope{ parent_scope };
    // Note: the order matters. Shaders, media, lights and models may depend on the transforms.
    //       Materials depend on shaders and media and lights.  Models depend on materials as well.
    ParseItems(node, "transforms", scope, &YamlSceneReader::ParseAndInsertTransform);
    ParseItems(node, "shaders", scope, &YamlSceneReader::ParseAndInsertItem<Shader>);
    ParseItems(node, "media", scope, &YamlSceneReader::ParseAndInsertItem<Medium>);
    ParseItems(node, "arealights", scope, &YamlSceneReader::ParseAndInsertItem<AreaEmitter>);
    ParseItems(node, "lights", scope, &YamlSceneReader::ParseAndInsertLight);
    ParseItems(node, "materials", scope, &YamlSceneReader::ParseAndInsertMaterial);
    ParseItems(node, "models", scope, &YamlSceneReader::ParseAndInsertModel);
    ParseItems(node, "scopes", scope, &YamlSceneReader::ParseScope);
    ParseView(node["view"]);
  }

  MaterialMap ParseMaterialMap(YAML::Node &node, const Scope &scope);

  Material LookupMaterial(const std::optional<std::string> &name, const MaterialMap &map, const Scope &scope) const;
  

  fs::path MakeFullPath(const YAML::Node &node, fs::path filename) const
  {
    try {
      return ctx.MakeFullPath(filename);
    }
    catch (const std::runtime_error &e)
    {
      throw MakeException(node, e.what());
    }
  }


  std::runtime_error MakeException(const YAML::Node &node, const std::string &msg) const
  {
    std::stringstream os;
    if (!ctx.GetFilename().empty())
      os << ctx.GetFilename() << ":";
    //os << util::Join("::", history.begin(), history.end(), [](auto &&x) { return x; }) << ":";
    os << "L" << node.Mark().line << " ";
    os << "\"" << msg << "\"";
    return std::runtime_error(os.str());
  }


  YAML::Node Pop(YAML::Node &node, const char* key) const
  {
    YAML::Node ret = TryPop(node, key);
    if (!ret.IsDefined())
      throw MakeException(node, fmt::format("Key error: {}", key));
    return ret;
  }

  YAML::Node TryPop(YAML::Node &node, const char* key) const
  {
    auto v = node[key];
    if (v)
    {
      bool ok = node.remove(key);
      assert(ok);
    }
    return v;
  }

  template<class T>
  T TryPop(YAML::Node &node, const char* key, const T &default_) const
  {
    auto v = TryPop(node, key);
    return v ? v.as<T>() : default_;
  }

  void ErrorOnRemainingKeys(YAML::Node &node) const
  {
    if (node.size())
    {
      auto msg = string{"Unused keys "}
               + util::Join(", ", node.begin(), node.end(), [](auto &&e) { return e.first; });
      throw MakeException(node, msg);
    }
  }

  template<class U>
  U* Lookup(const YAML::Node &node, const Scope &scope, const string &key) const
  {
    const std::optional<U*> x = scope.Lookup<U>(key);
    if (x)
      return *x;
    else
      throw MakeException(node, fmt::format("Key error: {}", key));
  }

  Color::SpectralN ParseSpectrum(YAML::Node &node, const string &prefix)
  {
    RGB ks_rgb{ 1._rgb };
    RGBScalar k{ 1._rgb };
    if (auto val = TryPop(node, prefix.c_str()); val)
    {
      auto tmp = val.as<Double3>();
      ks_rgb = RGB{ RGBScalar{tmp[0]},RGBScalar{tmp[1]},RGBScalar{tmp[2]} };
    }
    if (auto val = TryPop(node, (prefix + "_x").c_str()); val)
      k = RGBScalar{ val.as<double>() };
    return Color::RGBToSpectrum(k*ks_rgb);
  }

  std::shared_ptr<Texture> TryParseTexture(YAML::Node &node, const string &key)
  {
    auto texture_node = TryPop(node, key.c_str());
    if (texture_node)
    {
      auto path = MakeFullPath(texture_node, texture_node.as<std::string>());
      return std::make_shared<Texture>(path);
    }
    return { };
  }
};


void YamlSceneReader::ParseAndInsertTransform(const YAML::Node & original_node, Scope & scope)
{
  YAML::Node node = original_node;
  auto trafo = Transform::Identity();

  if (auto pos_node = TryPop(node, "pos"); pos_node)
  {
    trafo = Eigen::Translation3d(pos_node.as<Double3>());
  }
  else if (auto hpb_node = TryPop(node, "hpb"); hpb_node)
  {
    auto r = hpb_node.as<Double3>();
    if (!TryPop(node,"rad",false))
      r *= Pi / 180.;
    // The heading, pitch, bank convention assuming Y is up and Z is forward!
    trafo = trafo *
      Eigen::AngleAxisd(r[0], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(r[1], Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(r[2], Eigen::Vector3d::UnitZ());
  }
  if (auto rotaxis_node = TryPop(node, "rotaxis"); rotaxis_node)
  {
    auto r = rotaxis_node.as<Double3>();
    if (!TryPop(node, "rad", false))
      r *= Pi / 180.;
    trafo = trafo * Eigen::AngleAxisd(r.norm(), r.normalized());
  }
  if (auto scale_node = TryPop(node, "scale"); scale_node)
  {
    auto s = scale_node.as<Double3>();
    trafo = trafo * Eigen::Scaling(s);
  }
  ErrorOnRemainingKeys(node);
  scope.currentTransform = scope.currentTransform * trafo;
  std::cout << "Transform: t=\n" << scope.currentTransform.translation() << "\nr=\n" << scope.currentTransform.linear() << std::endl;
}


MaterialMap YamlSceneReader::ParseMaterialMap(YAML::Node & node, const Scope & scope)
{
  auto GetMaterial = [&node, this](const string &name, const Scope &scope) -> Material
  {
    if (auto it = scope.materials.find(name); it != scope.materials.end())
      return it->second;
    else
      throw MakeException(node, fmt::format("Key error {}", name));
  };

  MaterialMap mmap;
  auto default_material_name = TryPop<string>(node, "defaultmaterial", "default");
  mmap.default_material = GetMaterial(default_material_name, scope);
  auto mmap_list_node = TryPop(node, "materialmap");
  if (mmap_list_node)
  {
    for (auto it = mmap_list_node.begin(); it != mmap_list_node.end(); ++it)
    {
      auto target_name = it->second.as<string>();
      mmap.name_to_mat[it->first.as<string>()] = { target_name, GetMaterial(target_name, scope) };
    }
  }
  // TODO
  //mmap.use_object_names = TryPop<bool>(node, "use_object_names", false);
  return mmap;
}


void YamlSceneReader::ParseAndInsertModel(const YAML::Node & original_node, Scope & parent_scope)
{
  YAML::Node node = original_node;
  Scope scope{ parent_scope };
  if (node["file"])
  {
    auto filename = Pop(node, "file").as<string>();
    auto fullpath = MakeFullPath(node, filename);

    auto material_map = ParseMaterialMap(node, scope);
    auto material_assignment_by_object_names = TryPop(node, "material_assignment_by_object_names", false);
    auto material_getter = std::bind(
      &YamlSceneReader::LookupMaterial, 
      this, 
      std::placeholders::_1, 
      std::cref(material_map),
      std::cref(scope));
    assimp::Read(ctx.GetScene(), scope.currentTransform, material_getter, material_assignment_by_object_names, fullpath);
  }
  else if (node["sphere"])
  {
    Pop(node, "sphere");
    auto material_map = ParseMaterialMap(node, scope);
    auto pos = scope.currentTransform*Pop(node, "position").as<Double3>();
    auto rad = std::pow(scope.currentTransform.linear().determinant(), 1. / 3.)*
      Pop(node, "radius").as<double>();
    auto  sphere = Spheres();
    sphere.Append(pos.cast<float>(), rad);
    ctx.GetScene().Append(sphere, material_map.default_material);
  }
  ErrorOnRemainingKeys(node);
}

void YamlSceneReader::ParseView(const YAML::Node &original_node)
{
  if (!original_node)
    return;
  YAML::Node node = original_node;
  auto from = Pop(node, "from").as<Double3>();
  auto at = Pop(node, "at").as<Double3>();
  auto up = Pop(node, "up").as<Double3>();
  auto angle = Pop(node, "angle").as<double>();
  int resx = 640;
  int resy = 480;
  if (auto p = ctx.GetParams(); p)
  {
    resx = p->width;
    resy = p->height;
  }
  ctx.GetScene().camera = std::make_unique<PerspectiveCamera>(from, at - from, up, angle, resx, resy);
  ErrorOnRemainingKeys(node);
}


std::pair<string, materials::Medium*> YamlSceneReader::ParseItem(const YAML::Node & original_node, const Scope & scope, DispatchOverload<Medium>)
{
  YAML::Node node = original_node;
  auto class_ = Pop(node, "class").as<std::string>();
  auto name = TryPop(node, "name", string{});

  std::unique_ptr<materials::Medium> medium;
  if (class_ == "homogeneous")
  {
    auto absorb = Pop(node, "absorb");
    auto scatter = Pop(node, "scatter");
    if (absorb.IsScalar() && scatter.IsScalar())
    {
      medium = std::make_unique<materials::MonochromaticHomogeneousMedium>(
        scatter.as<double>(), absorb.as<double>(), scope.mediums.size());
    }
    else
    {
      medium = std::make_unique<materials::HomogeneousMedium>(
        Color::RGBToSpectrum(scatter.as<Double3>().cast<RGBScalar>()), 
        Color::RGBToSpectrum(absorb.as<Double3>().cast<RGBScalar>()),
        scope.mediums.size());
    }
  }
  else
    throw MakeException(node, fmt::format("Unknown class {}", class_));

  ErrorOnRemainingKeys(node);

  auto* p = medium.get();
  ctx.GetScene().media.push_back(std::move(medium));

  return { name, p };
}


std::pair<string, Shader*> YamlSceneReader::ParseItem(const YAML::Node & original_node, const Scope &, DispatchOverload<Shader>)
{
  YAML::Node node = original_node;

  auto name = TryPop(node, "name", std::string{});
  auto class_ = Pop(node, "class").as<std::string>();

  std::unique_ptr<Shader> shd;
  if (class_ == "speculartransmissivedielectric")
  {
    auto ior_ratio = Pop(node,"ior_ratio").as<double>();
    auto abbe_node = TryPop(node,"abbe_number");
    double ior_coeff = 0.;
    if (abbe_node)
    {
      double v = abbe_node.as<double>();
      // https://en.wikipedia.org/wiki/Abbe_number
      // v = (n(589)-1) / (n(486) - n(656))
      // Assuming that ior_ratio is the number for lambda=589nm.
      ior_coeff = (ior_ratio - 1) / v / (656 - 486);
    }
    shd = std::make_unique<SpecularTransmissiveDielectricShader>(ior_ratio, ior_coeff);
  }
  else if (class_ == "glossytransmissivedielectric")
  {
    auto ior_ratio = Pop(node, "ior_ratio").as<double>();
    auto alpha = Pop(node, "alpha").as<double>();
    auto texture = TryParseTexture(node, "alpha_texture");
    auto alpha_min = texture ? Pop(node, "alpha_min").as<double>() : alpha;
    shd = MakeGlossyTransmissiveDielectricShader(ior_ratio, alpha, alpha_min, texture);
  }
  else if (class_ == "glossy")
  {
    auto alpha = Pop(node, "alpha").as<double>();
    auto texture = TryParseTexture(node, "alpha_texture");
    auto reflection = ParseSpectrum(node, "rgb");
    shd = std::make_unique<MicrofacetShader>(reflection, alpha, texture);
  }
  else if (class_ == "diffuse")
  {
    auto reflection = ParseSpectrum(node, "rgb");
    auto texture = TryParseTexture(node, "texture");
    shd = std::make_unique<DiffuseShader>(reflection, texture);
  }
  else
    throw MakeException(node, fmt::format("Unkown class: {}", class_));

  if (auto val = TryPop(node, "prefer_path_tracing"); val)
    shd->prefer_path_tracing_over_photonmap = val.as<bool>();

  ErrorOnRemainingKeys(node);
  
  auto* p = shd.get();
  ctx.GetScene().shaders.emplace_back(std::move(shd));

  return { name, p };
}


std::pair<string, AreaEmitter*> YamlSceneReader::ParseItem(const YAML::Node &original_node, const Scope &scope, DispatchOverload<AreaEmitter>)
{
  YAML::Node node = original_node;

  auto name = TryPop(node, "name", std::string{});
  auto class_ = Pop(node, "class").as<std::string>();

  std::unique_ptr<AreaEmitter> light;
  {
    auto spectrum = ParseSpectrum(node, "rgb");

    if (class_ == "uniform")
    {
      light.reset(new UniformAreaLight(spectrum));
    }
    else if (class_ == "parallel")
    {
      light.reset(new ParallelAreaLight(spectrum));
    }
    else
      throw MakeException(node, fmt::format("Unkown class {}", class_));
  }

  ErrorOnRemainingKeys(node);

  auto* p = light.release(); // WARNING: releasing ownership
  // TODO: maybe keep track of area emitters, and free them properly on exit ...
  //ctx.GetScene().areaemitters.emplace_back(std::move(shd));
  return { name, p };
}


void YamlSceneReader::ParseAndInsertMaterial(const YAML::Node & original_node, Scope & scope)
{
  YAML::Node node = original_node;
  auto name = Pop(node, "name").as<std::string>();
  
  Material mat;
  
  auto shader_node = TryPop(node, "shader");
  auto medium_node = TryPop(node, "medium");
  auto outer_medium_node = TryPop(node, "outer_medium");
  auto emitter_node = TryPop(node, "arealight");
  
  if (shader_node)
    mat.shader = ParseReferenceOrInlined<Shader>(shader_node, "shader", scope);
  if (medium_node)
    mat.medium = ParseReferenceOrInlined<Medium>(medium_node, "medium", scope);
  if (outer_medium_node)
    mat.outer_medium = ParseReferenceOrInlined<Medium>(outer_medium_node, "outer_medium", scope);
  if (emitter_node)
    mat.emitter = ParseReferenceOrInlined<AreaEmitter>(emitter_node, "arealight", scope);
  scope.materials[name] = mat;
}


void YamlSceneReader::ParseAndInsertLight(const YAML::Node & original_node, Scope & scope)
{
  YAML::Node node = original_node;
  auto class_ = Pop(node, "class").as<std::string>();
  
  if (class_ == "point")
  {
    //auto col = TryPop<RGB>(node, "color", RGB::Ones());
    //auto intensity = TryPop<double>(node, "intensity", 1.);
    auto spectrum = ParseSpectrum(node, "rgb");
    auto pos = scope.currentTransform*Pop(node, "position").as<Double3>();
    auto light = std::make_unique<RadianceOrImportance::PointLight>(
      spectrum, pos);
    light->scene_index = isize(ctx.GetScene().lights);
    ctx.GetScene().lights.push_back(std::move(light));
  }
  else if (class_ == "sun")
  {
    auto dir = Normalized(Pop(node, "direction").as<Double3>());
    auto total_power = Pop(node, "total_power").as<double>();
    auto opening_angle = TryPop<double>(node, "opening_angle_deg", 0.25);
    auto light = std::make_unique<Sun>(total_power, -dir, opening_angle);
    ctx.GetScene().envlights.push_back(std::move(light));
  }
  else if (class_ == "directional")
  {
    auto dir = Normalized(Pop(node, "direction").as<Double3>());
    //auto col = TryPop<RGB>(node, "color", RGB::Ones());
    //auto intensity = TryPop<double>(node, "intensity", 1.);
    auto spectrum = ParseSpectrum(node, "rgb");
    auto light = std::make_unique<DistantDirectionalLight>(
      spectrum, -dir);
    ctx.GetScene().envlights.push_back(std::move(light));
  }
  else if (class_ == "dome")
  {
    auto dir = Normalized(Pop(node, "direction").as<Double3>());
    //auto col = TryPop<RGB>(node, "color", RGB::Ones());
    //auto intensity = TryPop<double>(node, "intensity", 1.);
    auto spectrum = ParseSpectrum(node, "rgb");
    auto light = std::make_unique<DistantDirectionalLight>(
      //intensity*Color::RGBToSpectrum(col),
      spectrum, dir);
    ctx.GetScene().envlights.push_back(std::move(light));
  }
  else if (class_ == "env")
  {
    auto dir = Normalized(Pop(node, "direction").as<Double3>());
    auto filename = Pop(node, "filename").as<string>();
    auto path = MakeFullPath(node, filename);
    auto tex = std::make_unique<Texture>(path);
    auto light = std::make_unique<EnvMapLight>(tex.get(), dir);
    ctx.GetScene().envlights.push_back(std::move(light));
    ctx.GetScene().textures.push_back(std::move(tex));
  }
  else
    throw MakeException(node, fmt::format("Unkown class {}", class_));

  ErrorOnRemainingKeys(node);
}


Material YamlSceneReader::LookupMaterial(const std::optional<std::string>& name, const MaterialMap & map, const Scope &scope) const
{
  if (!name)
    return map.default_material;
  if (auto it = map.name_to_mat.find(*name); it != map.name_to_mat.end())
  {
    fmt::print("Mapped material lookup by {} to {}\n", it->first, it->second.name);
    return it->second.mat;
  }
  if (auto it = scope.materials.find(*name); it != scope.materials.end())
  {
    fmt::print("Using material {}\n", it->first);
    return it->second;
  }
  fmt::print("Lookup of material name {} failed, using default\n", (*name));
  return map.default_material;
}


} // namespace scenereader



void Scene::ParseYAML(std::istream &is, RenderingParameters *render_params, const scenereader::fs::path &filename_hint)
{
  scenereader::YamlSceneReader reader{ *this, render_params, filename_hint.string() };
  scenereader::Scope scope;
  scenereader::AddDefaultMaterials(scope, *this);
  reader.Parse(is, scope);
  envlight = std::make_unique<RadianceOrImportance::TotalEnvironmentalRadianceField>(this->envlights);
}
#include "scene.hxx"
#include "camera.hxx"
#include "infiniteplane.hxx"
#include "shader.hxx"
#include "util.hxx"
#include "atmosphere.hxx"
#include "light.hxx"

#define LINESIZE 1000

#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <unordered_map>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

using namespace RadianceOrImportance;
namespace fs = boost::filesystem;


inline Double3 TransformNormal(const Eigen::Transform<double,3,Eigen::Affine> &trafo, const Double3 &v)
{
  return trafo.linear().inverse().transpose() * v;
}


// TODO: Rename this
template<class Thing>
class SymbolTable
{
  Thing currentThing;
  std::unordered_map<std::string, Thing> things;
  std::string name_of_this_table;
public:
  SymbolTable(const std::string &_name, const std::string &default_name, Thing default_thing) :
    currentThing(default_thing), name_of_this_table(_name)
  {
    things[default_name] = default_thing;
  }
  
  int size() const
  {
    return things.size();
  }
  
  void activate(const std::string &name)
  {
    auto it = things.find(name);
    if (it != things.end())
    {
      currentThing = it->second;
    }
    else
    {
      char buffer[1024];
      std::snprintf(buffer, 1024, "Error: %s %s not defined. Define it in the NFF file prior to referencing it.", this->name_of_this_table.c_str(), name.c_str());
      throw std::runtime_error(buffer);
    }
  }
  
  void set_and_activate(const char* name, Thing thing)
  {
    currentThing = thing;
    things[name] = thing;
  }
  
  Thing operator()() const
  {
    return currentThing;
  }
};



struct Scope
{
  SymbolTable<Shader*> shaders;
  SymbolTable<Medium*> mediums;
  SymbolTable<AreaEmitter*> areaemitters;
  Eigen::Transform<double,3,Eigen::Affine> currentTransform;
  
  Scope(const Scene &scene) :
    shaders("Shader", "invisible", scene.invisible_shader),
    mediums("Medium", "default", scene.empty_space_medium),
    areaemitters("AreaEmitter", "none", nullptr),
    currentTransform(decltype(currentTransform)::Identity())
  {
  }
};





class NFFParser
{
  Scene* scene;
  std::unordered_map<Material, MaterialIndex, Material::Hash> to_material_index;
  
  RenderingParameters *render_params;
  std::vector<fs::path> search_paths;
  fs::path    filename;
  std::string line;
  bool        line_stream_state;
  std::string peek_line;
  bool        peek_stream_state;
  std::istream &input;
  int lineno;
  friend class AssimpReader;
public:
  NFFParser(
      Scene* _scene,
      RenderingParameters *_render_params,
      std::istream &_is,
      const fs::path &_path_hint) :
    scene(_scene),
    render_params(_render_params),
    filename(_path_hint),
    input{_is},
    lineno{0}
  {
    for (int i = 0; i<scene->materials.size(); ++i)
      to_material_index[scene->materials[i]] = MaterialIndex(i);
    
    if (!filename.empty())
    {
      this->search_paths.emplace_back(
        filename.parent_path());
    }
    else
    {
      this->search_paths.emplace_back(fs::path(""));
    }
    if (render_params)
    {
      for (auto s: render_params->search_paths)
        this->search_paths.emplace_back(s);
    }
    line_stream_state = true;
    peek_stream_state = (bool)std::getline(input, peek_line);
  }

  void Parse(Scope &scope);
private:
  void ParseMesh(const char *filename, Scope &scope);
  MaterialIndex GetMaterialIndexOfCurrentParams(const Scope &scope);
  MaterialIndex MaterialInsertAndOrGetIndex(const Material &m);
  void InsertAndActivate(const char*, Scope &scope, std::unique_ptr<Medium> x);
  void InsertAndActivate(const char*, Scope &scope, std::unique_ptr<Shader> x);
  bool NextLine();
  std::runtime_error MakeException(const std::string &msg) const;
  fs::path MakeFullPath(const fs::path &filename) const;
};



void NFFParser::Parse(Scope &scope)
{
  char token[LINESIZE+1];
  while (NextLine())
  {
    if (line.empty())
      continue;
    
    if (line[0] == '#') // '#' : comment
      continue;
    
    // TODO: streams: https://stackoverflow.com/questions/1033207/what-should-i-use-instead-of-sscanf
    int numtokens = sscanf(line.c_str(),"%s",token);
    if (numtokens <= 0) // empty line, except for whitespaces
      continue; 
    
    
    if (!strcmp(token,"{")) {
      Scope child{scope};
      Parse(child);
      continue;
    }

    
    if (!strcmp(token,"}")) {
      break;
    }
    
    
    if (!strcmp(token, "transform"))
    {
      Double3 t, r, s;
      int n;
      n = std::sscanf(line.c_str(),"transform %lg %lg %lg %lg %lg %lg %lg %lg %lg\n",&t[0], &t[1], &t[2], &r[0], &r[1], &r[2], &s[0], &s[1], &s[2]);
      if (n == EOF) n = 0; // Number of arguments is actually zero.
      else if (n == 0) n = -1; // Failure.
      if (n !=0 && n != 3 && n != 6 && n != 9)
        throw std::runtime_error(strconcat("error in ", filename, " : ", line));
      decltype(scope.currentTransform) trafo;
      if (n == 0)
      {
        trafo = decltype(trafo)::Identity();
      }
      if (n >= 3)
      {
        trafo = Eigen::Translation3d(t);
      }
      if (n >= 6)
      {
        // The heading, pitch, bank convention assuming Y is up and Z is forward!
        trafo = trafo *
          Eigen::AngleAxisd(r[0], Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(r[1], Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(r[2], Eigen::Vector3d::UnitZ());
        if (n >= 9)
        {
          trafo = trafo * Eigen::Scaling(s);
        }
      }
      scope.currentTransform = trafo;
      std::cout << "Transform: t=\n" << scope.currentTransform.translation() << "\nr=\n" << scope.currentTransform.linear() << std::endl;
      continue;
    }
    
    /* camera */
    
    struct CommonCameraData
    {
      Double3 pos{NaN}, at{NaN}, up{NaN};
      int resX{-1}, resY{-1};
    };
    
    auto ParseCameraData = [this]() -> CommonCameraData
    {
      CommonCameraData cd;
      bool ok;
      ok = 3 == std::sscanf(line.c_str(),"from %lg %lg %lg\n",&cd.pos[0],&cd.pos[1],&cd.pos[2]);
      if (!ok) throw MakeException("Error");
      NextLine();
      ok = 3 == std::sscanf(line.c_str(),"at %lg %lg %lg\n",&cd.at[0],&cd.at[1],&cd.at[2]);
      if (!ok) throw MakeException("Error");
      NextLine();
      ok = 3 == std::sscanf(line.c_str(),"up %lg %lg %lg\n",&cd.up[0],&cd.up[1],&cd.up[2]);
      if (!ok) throw MakeException("Error");
      NextLine();
      ok = 2 == std::sscanf(line.c_str(),"resolution %d %d\n",&cd.resX,&cd.resY);
      if (!ok) throw MakeException("Error");
      NextLine();
      return cd;
    };
    
    auto MakeConsistentResolutionSettings = [this](CommonCameraData &cd)
    {
      if (render_params)
      {
        if (render_params->height > 0)
          cd.resY = render_params->height;
        else
          render_params->height = cd.resY;
        if (render_params->width > 0)
          cd.resX = render_params->width;
        else
          render_params->width = cd.resX;
      }
    };
    
    if (!strcmp(token, "vfisheye"))
    {
      NextLine();
      auto cd = ParseCameraData();
      MakeConsistentResolutionSettings(cd);
      scene->camera = std::make_unique<FisheyeHemisphereCamera>(cd.pos,cd.at-cd.pos,cd.up,cd.resX,cd.resY);
      continue;
    }
    
    if (!strcmp(token,"v"))
    {
      // FORMAT:
      //     v
      //     from %lg %lg %lg
      //     at %lg %lg %lg
      //     up %lg %lg %lg
      //     resolution %d %d
      //     angle %lg
      double angle{NaN};
      NextLine();
      auto cd = ParseCameraData();
      MakeConsistentResolutionSettings(cd);
      if (1 != std::sscanf(line.c_str(),"angle %lg\n",&angle)) 
        throw MakeException("Error");
      NextLine();
      scene->camera = std::make_unique<PerspectiveCamera>(cd.pos,cd.at-cd.pos,cd.up,angle,cd.resX,cd.resY);
      continue;
    }
    
    /* sphere */

    if (!strcmp(token,"s"))
    {
      Double3 pos;
      double rad;
      int n = sscanf(line.c_str(),"s %lg %lg %lg %lg",&pos[0],&pos[1],&pos[2],&rad);
      if (n == 4)
      {
          pos = scope.currentTransform*pos;
          MaterialIndex current_material_index = GetMaterialIndexOfCurrentParams(scope);
          scene->spheres->Append(pos.cast<float>(), rad, current_material_index);
      }
      else throw MakeException("Error");
      continue;
    }

    /* polygon (with normals and uv) */
    if (!strcmp(token,"p")) 
    {
      int num_vertices;
      sscanf(line.c_str(),"p %d",&num_vertices);
      
      if (num_vertices < 3)
        throw MakeException("Polygon must be specified with at least 3 vertices");
      
      Mesh mesh(num_vertices-2, num_vertices);
      bool must_compute_normal = false;
      
      for (int i=0;i<num_vertices;i++)
      {
        Double3 v{0.};
        Double3 n{0.};
        Double2 uv{0.};
        if (!NextLine())
          throw MakeException("Cannot read specified number of vertices");
        int n_read = sscanf(line.c_str(),"%lg %lg %lg %lg %lg %lg %lg %lg\n", 
                            &v[0],&v[1],&v[2], &n[0],&n[1],&n[2], &uv[0],&uv[1]);
        if (n_read < 3)
          throw MakeException("Cannot vertex coordinates");
        
        // Compute normals from vertices if one vertex has no or invalid normal specification.
        must_compute_normal |= (n_read < 6);
        v = scope.currentTransform * v;
        n = TransformNormal(scope.currentTransform, n);
        mesh.vertices.row(i) = v.cast<float>();
        mesh.normals.row(i) = n.cast<float>();
        mesh.uvs.row(i) = uv.cast<float>();
      }

      MaterialIndex current_material_index = GetMaterialIndexOfCurrentParams(scope);
      
      for (int i=0; i<num_vertices-2; i++) 
      {
        mesh.vert_indices(i, 0) = 0;
        mesh.vert_indices(i, 1) = i+1;
        mesh.vert_indices(i, 2) = i+2;
        mesh.material_indices[i] = current_material_index;
      }

      if (must_compute_normal)
        mesh.MakeFlatNormals();
      
      // TODO: This is O(n*n)! Collect all meshes and concatenate them all at once in the end.
      scene->Append(mesh);
      continue;
    }
    
    
    /* shader parameters */
    if (!strcmp(token, "shader"))
    {
      char name[LINESIZE];
      int num = std::sscanf(line.c_str(), "shader %s\n", name);
      if(num==1)
      {
        scope.shaders.activate(name);
      }
      else throw MakeException("shader directive needs name of the shader.");
      continue;
    }
   
    
    auto MaybeReadTexture = [this](const char *identifier) -> std::unique_ptr<Texture>
    {
      if (startswith(peek_line, identifier))
      {
        NextLine();
        std::string format = strconcat(identifier, " %s\n");
        char buffer [LINESIZE];
        int num = std::sscanf(line.c_str(), format.c_str(), buffer);
        if (num == 1)
        {
          auto path = MakeFullPath(buffer);
          return std::make_unique<Texture>(path);
        }
        else 
          throw MakeException("Error");
      }
      else
        return std::unique_ptr<Texture>(nullptr);
    };
    
    
    if (!strcmp(token,"diffuse"))
    {
      RGB rgb;
      RGBScalar kd;
      char name[LINESIZE];
      int num = std::sscanf(line.c_str(),"diffuse %s %lg %lg %lg %lg\n",name, &rgb[0],&rgb[1],&rgb[2],&kd);
      if (num == 5)
      {
        std::unique_ptr<Texture> diffuse_texture = MaybeReadTexture("diffusetexture");
        InsertAndActivate(name, scope,
          std::make_unique<DiffuseShader>(
            Color::RGBToSpectrum(kd * rgb), std::move(diffuse_texture)));
      }
      else throw MakeException("Error");
      continue;
    }
    
    if (!strcmp(token,"specularreflective"))
    {
      RGB rgb;
      RGBScalar k;
      char name[LINESIZE];
      int num = std::sscanf(line.c_str(),"specularreflective %s %lg %lg %lg %lg\n",name, &rgb[0],&rgb[1],&rgb[2],&k);
      if (num == 5)
      {
        InsertAndActivate(name, scope,
          std::make_unique<SpecularReflectiveShader>(
            Color::RGBToSpectrum(k * rgb)
          ));
      }
      else throw MakeException("Error");
      continue;
    }
    
    if (!strcmp(token, "speculartransmissivedielectric"))
    {
      double ior_ratio = 1.;
      char name[LINESIZE];
      int num = std::sscanf(line.c_str(), "speculartransmissivedielectric %s %lg\n", name, &ior_ratio);
      if (num == 2)
      {
        InsertAndActivate(name, scope,
          std::make_unique<SpecularTransmissiveDielectricShader>(ior_ratio));
      }
      else throw MakeException("Error");
      continue;
    }
    
    if (!strcmp(token, "speculardensedielectric"))
    {
      RGB diffuse_coeff;
      double specular_coeff;
      char name[LINESIZE];
      int num = std::sscanf(line.c_str(), "speculardensedielectric %s %lg %lg %lg %lg\n", name, &diffuse_coeff[0], &diffuse_coeff[1], &diffuse_coeff[2], &specular_coeff);
      if (num == 5)
      {
        std::unique_ptr<Texture> diffuse_texture = MaybeReadTexture("diffusetexture");
        InsertAndActivate(name, scope,
          std::make_unique<SpecularDenseDielectricShader>(
            specular_coeff,
            Color::RGBToSpectrum(diffuse_coeff),
            std::move(diffuse_texture)));
      }
      else throw MakeException("Error");
      continue;
    }
    
    if (!strcmp(token,"glossy"))
    {
      RGBScalar k;
      double phong_exponent;
      RGB kd_rgb, ks_rgb;
      char name[LINESIZE];
      
      int num = std::sscanf(line.c_str(),"glossy %s %lg %lg %lg %lg %lg\n",name,&ks_rgb[0], &ks_rgb[1], &ks_rgb[2], &k, &phong_exponent);
      if(num == 6)
      {
        std::unique_ptr<Texture> glossy_texture = MaybeReadTexture("exponenttexture");
        InsertAndActivate(name, scope, 
          std::make_unique<MicrofacetShader>(
            Color::RGBToSpectrum(k*ks_rgb),
            phong_exponent,
            std::move(glossy_texture)));
      }
      else throw MakeException("Error");
      continue;
    }
    
    
    auto MaybeReadPF = [this]() -> std::unique_ptr<PhaseFunctions::PhaseFunction>
    {
      if (startswith(this->peek_line, "pf "))
      {
        this->NextLine();
        double g;
        char name[LINESIZE];
        int num = std::sscanf(this->line.c_str(),"pf %s %lg\n",name, &g);
        if (num > 0)
        {
          if (!strcmp(name, "rayleigh"))
          {
            return std::make_unique<PhaseFunctions::Rayleigh>();
          }
          else if (!strcmp(name, "henleygreenstein") && num>1)
          {
            return std::make_unique<PhaseFunctions::HenleyGreenstein>(g);
          }
        }
        throw MakeException("Error");
      }
      else
        return std::unique_ptr<PhaseFunctions::PhaseFunction>{nullptr};
    };
    
    
    if (!strcmp(token, "medium"))
    {
      RGBScalar buffer[6];
      char name[LINESIZE];
      int num = std::sscanf(line.c_str(), "medium %s %lg %lg %lg %lg %lg %lg\n", name, &buffer[0], &buffer[1], &buffer[2], &buffer[3], &buffer[4], &buffer[5]);
      if(num == 1)
      {
        scope.mediums.activate(name);
      }
      else if(num == 3)
      {
        auto medium = std::make_unique<MonochromaticHomogeneousMedium>(
          value(buffer[0]), 
          value(buffer[1]), 
          scope.mediums.size());
        auto pf = MaybeReadPF();
        if (pf)
          medium->phasefunction = std::move(pf);
        InsertAndActivate(name, scope, std::move(medium));
      }
      else if(num == 7)
      {
        auto medium = std::make_unique<HomogeneousMedium>(
          Color::RGBToSpectrum({buffer[0], buffer[1], buffer[2]}),
          Color::RGBToSpectrum({buffer[3], buffer[4], buffer[5]}), 
          scope.mediums.size());
        auto pf = MaybeReadPF();
        if (pf)
          medium->phasefunction = std::move(pf);
        InsertAndActivate(name, scope, std::move(medium));
      }
      else throw MakeException("Error");
      continue;
    }


    if (!strcmp(token, "simpleatmosphere"))
    {
      Double3 planet_center;
      double radius;
      char name[LINESIZE];
      int num = std::sscanf(line.c_str(), "simpleatmosphere %s %lg %lg %lg %lg\n", name, &planet_center[0], &planet_center[1], &planet_center[2], &radius);
      if (num==1)
      {
        scope.mediums.activate(name);
      }
      else if(num == 5)
      {
        auto medium = Atmosphere::MakeSimple(planet_center, radius, scope.mediums.size());
        InsertAndActivate(
          name, scope, std::move(medium));
      }
      else
      {
        throw MakeException("Error");
      }
      continue;
    }

    if (!strcmp(token, "tabulatedatmosphere"))
    {
      Double3 planet_center;
      double radius;
      char name[LINESIZE];
      char datafile[LINESIZE];
      int num = std::sscanf(line.c_str(), "tabulatedatmosphere %s %lg %lg %lg %lg %s\n", name, &planet_center[0], &planet_center[1], &planet_center[2], &radius, datafile);
      if (num==1)
      {
        scope.mediums.activate(name);
      }
      else if(num == 6)
      {
        auto full_datafile_path = MakeFullPath(datafile);
        auto medium = Atmosphere::MakeTabulated(planet_center, radius, full_datafile_path.string(), scope.mediums.size());
        InsertAndActivate(
          name, scope, std::move(medium));
      }
      else
      {
        throw MakeException("Error");
      }
      continue;
    }
    
    
    if (!strcmp(token, "lsun"))
    {
      Double3 dir_out;
      double total_power, opening_angle;
      int num = std::sscanf(line.c_str(),"lsun %lg %lg %lg %lg %lg",
          &dir_out[0],&dir_out[1],&dir_out[2], &total_power, &opening_angle);
      Normalize(dir_out);
      if (num == 4)
      {
        // "The Sun is seen from Earth at an average angular diameter of about 9.35×10−3 radians."
        // https://en.wikipedia.org/wiki/Solid_angle
        opening_angle = 0.26;
        num = 5;
      }
      if (num == 5)
      {
        scene->envlights.push_back(std::make_unique<Sun>(total_power, dir_out, opening_angle));
      }
      else
      {
        throw MakeException("Error");
      }
      continue;
    }
    
  
    if (!strcmp(token, "lddirection"))
    {
      Double3 dir_out;
      RGB col;
      int num = sscanf(line.c_str(),"lddirection %lg %lg %lg %lg %lg %lg",
          &dir_out[0],&dir_out[1],&dir_out[2],
          &col[0],&col[1],&col[2]);
      Normalize(dir_out);
      if (num == 6)
      {
        scene->envlights.push_back(std::make_unique<DistantDirectionalLight>(
          Color::RGBToSpectrum(col), 
          dir_out
        ));
      }
      else
      {
        throw MakeException("Error");
      }
      continue;
    }
  
  
    if (!strcmp(token, "lddome"))
    {
      Double3 dir_up;
      RGB col;
      int num = sscanf(line.c_str(),"lddome %lg %lg %lg %lg %lg %lg",
          &dir_up[0],&dir_up[1],&dir_up[2],
          &col[0],&col[1],&col[2]);
      Normalize(dir_up);
      if (num == 6)
      {
        scene->envlights.push_back(std::make_unique<DistantDomeLight>(
          Color::RGBToSpectrum(col), 
          dir_up
        ));
      }
      else
      {
        throw MakeException("Error");
      }
      continue;
    }
  
  
  if (!strcmp(token, "larea"))
  {
    char name[LINESIZE];
    char type[LINESIZE];
    int num = std::sscanf(line.c_str(), "larea %s %s", name, type);
    if (num == 1)
    {
      scope.areaemitters.activate(name);
    }
    else if (num == 2)
    {
      if (!strcmp(type, "uniform") || !strcmp(type, "parallel"))
      {
        RGB col;
        double area_power_density;
        char name[LINESIZE];
        int num = std::sscanf(line.c_str(), "larea %s %s %lg %lg %lg %lg",
                              name, type, &col[0], &col[1], &col[2], &area_power_density);
        if (num == 6)
        {
          if (!strcmp(type, "uniform"))
          {
            scope.areaemitters.set_and_activate(
              name, 
              new UniformAreaLight(area_power_density*Color::RGBToSpectrum(col))
            );
          }
          else
          {
            scope.areaemitters.set_and_activate(
              name,
              new ParallelAreaLight(area_power_density*Color::RGBToSpectrum(col))
            );
          }
        }
        else
        {
          throw MakeException("Error");
        }
      }
    }
    else
    {
      throw MakeException("Error");
    }
    continue;
  }
  
    /* lightsource */
  if (!strcmp(token,"l")) 
	{
		Double3 pos;
    RGB col;
    RGBScalar col_multiplier;
		int num = sscanf(line.c_str(),"l %lg %lg %lg %lg %lg %lg %lg",
			   &pos[0],&pos[1],&pos[2],
			   &col[0],&col[1],&col[2], &col_multiplier);
		if (num == 3) {
			// light source with position only
			col = RGB::Constant(1._rgb);
			scene->lights.push_back(std::make_unique<RadianceOrImportance::PointLight>(
        Color::RGBToSpectrum(col),
        pos
      ));
		} else if (num == 7) {
			// light source with position and color
			scene->lights.push_back(std::make_unique<PointLight>(
        Color::RGBToSpectrum(col_multiplier*col),
        pos
      ));	
		} else {
      throw MakeException("Error");
		}
		continue;
    }

	
// 	if(!strcmp(token,"sl"))
// 	{
// 		Double3 pos,dir,col;
// 		double min=0,max=10;
// 		int num = sscanf(line.c_str(),"sl %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg",
// 				&pos[0],&pos[1],&pos[2],&dir[0],&dir[1],&dir[2],&col[0],&col[1],&col[2],&min,&max); 
// 		if(num == 11) {
// 			scene->AddLight(std::make_unique<SpotLight>(col,pos,dir,min,max));
// 		}
// 		else {
// 			std::cout << "error in "<<filename<<" : " << line << std::endl;
// 		}
// 		continue;
// 	}
	

  /* include new NFF file */
  if (!strcmp(token,"include"))
  {
    char name[LINESIZE];
    int num = std::sscanf(line.c_str(),"include %s\n", name);
    if (num != 1)
    {
      throw MakeException("Unable to parse include line");
    }
    else
    {
      auto fullpath = MakeFullPath(name);
      std::cout << "including file " << fullpath << std::endl;
      std::ifstream is(fullpath.string());
      if (!is.good())
      {
        throw std::runtime_error(strconcat("Could not open input file", fullpath));
      }
      // Using this scope.
      NFFParser(scene, render_params, is, fullpath).Parse(scope);
    }
    continue;
  }

	
	if (!strcmp(token, "m"))
  {
    char meshfile[LINESIZE];
    int num = sscanf(line.c_str(), "m %s", meshfile);
    if (num == 1)
    {
      auto fullpath = MakeFullPath(meshfile);
      ParseMesh(fullpath.c_str(), scope);
    }
    else
    {
      throw MakeException("Error");
    }
    continue;
  }

    throw MakeException("Unkown directive");
  }
};


// Example see: https://github.com/assimp/assimp/blob/master/samples/SimpleOpenGL/Sample_SimpleOpenGL.c
class AssimpReader
{
  struct NodeRef
  {
    NodeRef(aiNode* _node, const aiMatrix4x4 &_mTrafoParent)
      : node(_node), local_to_world(_mTrafoParent * _node->mTransformation)
    {}
    aiNode* node;
    aiMatrix4x4 local_to_world;
  };
public:
  AssimpReader(NFFParser &parser, Scope &scope)
    : scene{*parser.scene},
      parser{parser},
      outer_scope{scope}
  {
  }
  
  void Read(const char *filename)
  {
    std::printf("Reading Mesh: %s\n", filename);
    this->aiscene = aiImportFile(filename, 0);
    
    if (!aiscene)
    {
      char buffer[1024];
      std::snprintf(buffer, 1024, "Error: could not load file %s. because: %s", filename, aiGetErrorString());
      throw std::runtime_error(buffer);
    }
    
    std::vector<NodeRef> nodestack{ {aiscene->mRootNode, aiMatrix4x4{}} };
    while (!nodestack.empty())
    {
      auto ndref = nodestack.back();
      nodestack.pop_back();
      for (unsigned int i = 0; i< ndref.node->mNumChildren; ++i)
      {
        nodestack.push_back({ndref.node->mChildren[i], ndref.local_to_world});
      }
      
      ReadNode(ndref);
    }
    
    aiReleaseImport(this->aiscene);
  }
  
private:
  void ReadNode(const NodeRef &ndref)
  {
    const auto *nd = ndref.node;
    for (unsigned int mesh_idx = 0; mesh_idx < nd->mNumMeshes; ++mesh_idx)
    {
      const aiMesh* mesh = aiscene->mMeshes[nd->mMeshes[mesh_idx]];
      std::printf("Mesh %i (%s), mat_idx=%i\n", mesh_idx, mesh->mName.C_Str(), mesh->mMaterialIndex);
      Scope scope{outer_scope};
      DealWithTheMaterialAssignment(scope, mesh);
      ReadMesh(scope, mesh, ndref);
    }
  }
    
  void ReadMesh(Scope &scope, const aiMesh* aimesh, const NodeRef &ndref)
  {
    auto m = ndref.local_to_world;
    auto m_linear = aiMatrix3x3(m);
    bool hasuv = aimesh->GetNumUVChannels()>0;
    bool hasnormals = aimesh->HasNormals();
    
    std::vector<UInt3> vert_indices; vert_indices.reserve(1024);
    for (unsigned int face_idx = 0; face_idx < aimesh->mNumFaces; ++face_idx)
    {
      const aiFace* face = &aimesh->mFaces[face_idx];
      for (int i=2; i<face->mNumIndices; i++)
      {
        vert_indices.push_back(
          UInt3(face->mIndices[0],
                 face->mIndices[i-1],
                 face->mIndices[i]));
      }
    }
    
    Mesh mesh(vert_indices.size(), aimesh->mNumVertices);
    for (int i=0; i<vert_indices.size(); ++i)
      mesh.vert_indices.row(i) = vert_indices[i];
    
    for (int i=0; i<aimesh->mNumVertices; ++i)
    {
      Double3 v = 
            scope.currentTransform *
              aiVector3_to_myvector(m*aimesh->mVertices[i]);
      assert(v.allFinite());
      mesh.vertices.row(i) = v.cast<float>();
    }

    if (hasnormals)
    {
      for (int i=0; i<aimesh->mNumVertices; ++i)
      {
        Double3 n =
          TransformNormal(scope.currentTransform,
            aiVector3_to_myvector(m_linear*aimesh->mNormals[i]));
        assert(n.allFinite());
        mesh.normals.row(i) = n.cast<float>();
      }
    }
    else
      mesh.MakeFlatNormals();
    
    if (hasuv)
    {
      for (int i=0; i<aimesh->mNumVertices; ++i)
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
    
    MaterialIndex mat_idx = parser.GetMaterialIndexOfCurrentParams(scope);
    for (int i=0; i<mesh.NumTriangles(); ++i)
      mesh.material_indices[i] = mat_idx;
    
    scene.Append(mesh);
    
    // TODO: delete me!
    for (int tri_index = 0; tri_index < mesh.NumTriangles(); ++tri_index)
    {
      int a = mesh.vert_indices(tri_index, 0);
      int b = mesh.vert_indices(tri_index, 1);
      int c = mesh.vert_indices(tri_index, 2);

      Double2 uv[3];
      Double3 no[3];

      no[0] = mesh.normals.row(a).cast<double>();
      no[1] = mesh.normals.row(b).cast<double>();
      no[2] = mesh.normals.row(c).cast<double>();

      uv[0] = mesh.uvs.row(a).cast<double>();
      uv[1] = mesh.uvs.row(b).cast<double>();
      uv[2] = mesh.uvs.row(c).cast<double>();
    }
  }
  
  
  void DealWithTheMaterialAssignment(Scope &scope, const aiMesh* mesh)
  {
    if (mesh->mMaterialIndex < aiscene->mNumMaterials)
    {
      SetCurrentShader(scope, aiscene->mMaterials[mesh->mMaterialIndex]);
    }
  }
  
  
  void SetCurrentShader(Scope &scope, const aiMaterial *mat)
  {
//     for (int prop_idx = 0; prop_idx < mat->mNumProperties; ++prop_idx)
//     {
//       const auto *prop = mat->mProperties[prop_idx];
//       aiString name;
//       
//       std::printf("Mat %p key[%i/%s]\n", (void*)mat, prop_idx, prop->mKey.C_Str());
//     }
    aiString ainame;
    mat->Get(AI_MATKEY_NAME,ainame);
    auto name = std::string(ainame.C_Str());
    if (name != "DefaultMaterial")
      scope.shaders.activate(name);
  }

private:
  const aiScene *aiscene = { nullptr };
  Scene &scene;
  NFFParser &parser;
  Scope &outer_scope;

  inline Double3 aiVector3_to_myvector(const aiVector3D &v)
  {
    return Double3{v[0], v[1], v[2]};
  }
};


void NFFParser::ParseMesh(const char* filename, Scope &scope)
{
  AssimpReader(*this, scope).Read(filename);
}


void Scene::ParseNFF(const fs::path &filename, RenderingParameters *render_params)
{
  std::ifstream is(filename.string());
  if (!is.good())
  {
    throw std::runtime_error(strconcat("Could not open input file", filename));
  }
  Scope scope{*this};
  NFFParser(this, render_params, is, filename).Parse(scope);
}


void Scene::ParseNFFString(const std::string &scenestr, RenderingParameters *render_params)
{
  std::istringstream is(scenestr);
  ParseNFF(is, render_params);
}


void Scene::ParseNFF(std::istream &is, RenderingParameters *render_params)
{
  Scope scope{*this};
  NFFParser(this, render_params, is, std::string()).Parse(scope);
}


bool NFFParser::NextLine()
{
  line = peek_line;
  line_stream_state = peek_stream_state;
  ++lineno;
  peek_stream_state = (bool)std::getline(input, peek_line);
  return line_stream_state;
}


MaterialIndex NFFParser::MaterialInsertAndOrGetIndex(const Material& m)
{
  return GetOrInsertFromFactory(to_material_index, m, [this, &m]() {
    scene->materials.push_back(m);
    return MaterialIndex(scene->materials.size()-1);
  });
}


void NFFParser::InsertAndActivate(const char* name, Scope& scope, std::unique_ptr< Medium > x)
{
  scope.mediums.set_and_activate(name, x.get());
  scene->media.emplace_back(std::move(x));
}


void NFFParser::InsertAndActivate(const char* name, Scope& scope, std::unique_ptr< Shader > x)
{
  scope.shaders.set_and_activate(name, x.get());
  scene->shaders.emplace_back(std::move(x));
}


MaterialIndex NFFParser::GetMaterialIndexOfCurrentParams(const Scope &scope)
{
  Material mat{scope.shaders(), scope.mediums(), scope.areaemitters()};
  return MaterialInsertAndOrGetIndex(mat);
}


fs::path NFFParser::MakeFullPath(const fs::path &filename) const
{
  if (filename.is_relative())
  {
    for (auto parent_path : this->search_paths)
    {
      auto trial = parent_path / filename;
      if (fs::exists(trial))
        return trial;
    }
    throw MakeException("Cannot find a file in the search paths matching the name "+filename.string());
  }
  else
    return filename;
}


std::runtime_error NFFParser::MakeException(const std::string &msg) const
{
  std::stringstream os;
  if (!filename.empty())
    os << filename << ":";
  os << lineno << ": " << msg << " [" << line << "]";
  return std::runtime_error(os.str());
}


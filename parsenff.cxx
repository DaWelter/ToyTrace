#include "scene.hxx"
#include "perspectivecamera.hxx"
#include "sphere.hxx"
#include "infiniteplane.hxx"
#include "triangle.hxx"
#include "shader.hxx"
#include "util.hxx"
#include "atmosphere.hxx"

#define LINESIZE 1000

#include <libgen.h> // unix specific. Need for dirname.
#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <unordered_map>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// TODO: Rename this
template<class Thing>
class CurrentThing
{
  Thing* currentThing;
  std::unordered_map<std::string, Thing*> things;
  std::string name;
public:
  CurrentThing(const std::string &_name, const std::string &default_name, Thing* default_thing) :
    name(_name), currentThing(default_thing)
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
      std::snprintf(buffer, 1024, "Error: %s %s not defined. Define it in the NFF file prior to referencing it.", this->name.c_str(), name.c_str());
      throw std::runtime_error(buffer);
    }
  }
  
  void set_and_activate(const char* name, Thing* thing)
  {
    currentThing = thing;
    things[name] = thing;
  }
  
  Thing* operator()() const
  {
    return currentThing;
  }
};



class NFFParser
{
  Scene* scene;
  CurrentThing<Shader> shaders;
  CurrentThing<Medium> mediums;
  Eigen::Transform<double,3,Eigen::Affine> currentTransform;
  std::string directory;
  std::string filename;
  std::string line;
  bool        line_stream_state;
  std::string peek_line;
  bool        peek_stream_state;
  std::istream &input;
  int lineno;
  RenderingParameters *render_params;
  friend class AssimpReader;
public:
  NFFParser(
      Scene* _scene,
      RenderingParameters *_render_params,
      std::istream &_is,
      const std::string &_path_hint)
      :
    scene(_scene),
    shaders("Shader", "invisible", new InvisibleShader()),
    mediums("Medium", "default", new VacuumMedium()),
    render_params(_render_params),
    input{_is},
    lineno{0},
    filename(_path_hint)
  {
    shaders.set_and_activate("default", new DiffuseShader(Spectral3(0.8, 0.8, 0.8), nullptr));
    currentTransform = decltype(currentTransform)::Identity();
    if (!filename.empty())
    {
      auto *dirnamec_storage = strdup(filename.c_str()); // modifies its argument.
      auto *dirnamec = ::dirname(dirnamec_storage);
      directory.assign(dirnamec);
      free(dirnamec_storage);
    }
    line_stream_state = true;
    peek_stream_state = (bool)std::getline(input, peek_line);
  }

  void Parse();
private:
  void ParseMesh(const char *filename);

  bool NextLine();
  std::runtime_error MakeException(const std::string &msg);
  std::string MakeFullPath(const std::string &filename) const;

  void AssignCurrentMaterialParams(Primitive &primitive);
  Double3 ApplyTransform(const Double3 &p) const;
  Double3 ApplyTransformNormal(const Double3 &v) const;
};


Double3 NFFParser::ApplyTransform(const Double3 &p) const
{
  return currentTransform * p;
}


Double3 NFFParser::ApplyTransformNormal(const Double3 &v) const
{
  return Normalized(currentTransform.linear() * v);
}


void NFFParser::AssignCurrentMaterialParams(Primitive& primitive)
{
  primitive.shader = shaders();
  primitive.medium = mediums();
  assert(primitive.shader && primitive.medium);
}


std::string NFFParser::MakeFullPath(const std::string &filename) const
{
  // Assuming we get a relative path as input.
  std::string fullpath = directory + (directory.empty() ? "" : "/") + filename;
  return fullpath;
}


std::runtime_error NFFParser::MakeException(const std::string &msg)
{
  std::stringstream os;
  if (!filename.empty())
    os << filename << ":";
  os << lineno << ": " << msg << " [" << line << "]";
  return std::runtime_error(os.str());
}


bool NFFParser::NextLine()
{
  line = peek_line;
  line_stream_state = peek_stream_state;
  ++lineno;
  peek_stream_state = (bool)std::getline(input, peek_line);
  return line_stream_state;
}


//void NFFParser::Parse(const std::string &_filename)
//{
//  filename = _filename;
//  std::ifstream is(filename);
//  if (!is.good())
//  {
//    throw std::runtime_error(strconcat("Could not open input file", filename));
//  }
//}


void NFFParser::Parse()
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
    
    if (!strcmp(token,"begin_hierarchy")) {
      // Accepted for legacy reasons
      continue;
    }

    
    if (!strcmp(token,"end_hierarchy")) {
      // Accepted for legacy reasons
      continue;
    }
    
    if (!strcmp(token, "transform"))
    {
      Double3 t, r, s;
      int n = std::sscanf(line.c_str(),"transform %lg %lg %lg %lg %lg %lg %lg %lg %lg\n",&t[0], &t[1], &t[2], &r[0], &r[1], &r[2], &s[0], &s[1], &s[2]);
      if (n != 3 && n != 6 && n != 9)
        throw std::runtime_error(strconcat("error in ", filename, " : ", line));
      decltype(currentTransform) trafo;
      trafo = Eigen::Translation3d(t);
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
      currentTransform = trafo;
      std::cout << "Transform: t=\n" << currentTransform.translation() << "\nr=\n" << currentTransform.linear() << std::endl;
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
      scene->SetCamera<FisheyeHemisphereCamera>(cd.pos,cd.at-cd.pos,cd.up,cd.resX,cd.resY);
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
      scene->SetCamera<PerspectiveCamera>(cd.pos,cd.at-cd.pos,cd.up,angle,cd.resX,cd.resY);
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
          AssignCurrentMaterialParams(
            scene->AddPrimitive<Sphere>(pos,rad));
      }
      else throw MakeException("Error");
      continue;
    }

/* polygon (with normals and uv) */
    if (!strcmp(token,"tpp")) 
	{
		int vertices;
    sscanf(line.c_str(),"tpp %d",&vertices);
		Double3 *vertex = new Double3[vertices];
		Double3 *normal = new Double3[vertices];
		Double3 *uv     = new Double3[vertices];

    for (int i=0;i<vertices;i++)
    {
      if (!NextLine() ||
          sscanf(line.c_str(),"%lg %lg %lg %lg %lg %lg %lg %lg %lg\n",
        &vertex[i][0],&vertex[i][1],&vertex[i][2],
        &normal[i][0],&normal[i][1],&normal[i][2],
        &uv[i][0],&uv[i][1],&uv[i][2]) < 9)
        throw MakeException("Error reading TexturedSmoothTriangle");
      vertex[i] = ApplyTransform(vertex[i]);
      normal[i] = ApplyTransformNormal(normal[i]);
		}

    for (int i=2;i<vertices;i++) {
      AssignCurrentMaterialParams(
        scene->AddPrimitive<TexturedSmoothTriangle>(
            vertex[0],
            vertex[i-1],
            vertex[i],
            normal[0],
            normal[i-1],
            normal[i],
            uv[0],
            uv[i-1],
            uv[i]));
		}
		delete[] vertex;
		delete[] normal;
    delete[] uv;
		continue;
    }

   
    /* polygon */

    if (!strcmp(token,"p")) {
		int vertices;
		sscanf(line.c_str(),"p %d",&vertices);
		Double3 *vertex = new Double3[vertices];
    for (int i=0;i<vertices;i++)
    {
      if (!NextLine() ||
          sscanf(line.c_str(),"%lg %lg %lg\n",&vertex[i][0],&vertex[i][1],&vertex[i][2]) < 3)
        throw MakeException("Error reading Triangle");
      vertex[i] = ApplyTransform(vertex[i]);
		}

    for (int i=2;i<vertices;i++) {
			AssignCurrentMaterialParams(
        scene->AddPrimitive<Triangle>(
          vertex[0],
					vertex[i-1],
					vertex[i]));
			}
		delete[] vertex;
		continue;
    }
    
    /* shader parameters */
    if (!strcmp(token, "shader"))
    {
      char name[LINESIZE];
      int num = std::sscanf(line.c_str(), "shader %s\n", name);
      if(num==1)
      {
        shaders.activate(name);
      }
      else throw MakeException("shader directive needs name of the shader.");
      continue;
    }
    
    
    if (!strcmp(token,"diffuse"))
    {
      Spectral3 rgb;
      double kd;
      char name[LINESIZE];
      int num = std::sscanf(line.c_str(),"diffuse %s %lg %lg %lg %lg\n",name, &rgb[0],&rgb[1],&rgb[2],&kd);
      if (num == 5)
      {
        std::unique_ptr<Texture> diffuse_texture;
        if (startswith(peek_line, "diffusetexture"))
        {
          NextLine();
          char buffer [LINESIZE];
          num = std::sscanf(line.c_str(), "diffusetexture %s\n", buffer);
          if (num == 1)
          {
            auto path = MakeFullPath(buffer);
            diffuse_texture = std::make_unique<Texture>(path);
          }
          else throw MakeException("Error");
        }
        shaders.set_and_activate(name, 
          new DiffuseShader(kd * rgb, std::move(diffuse_texture)));
      }
      else throw MakeException("Error");
      continue;
    }
    
    if (!strcmp(token,"specularreflective"))
    {
      Spectral3 rgb;
      double k;
      char name[LINESIZE];
      int num = std::sscanf(line.c_str(),"specularreflective %s %lg %lg %lg %lg\n",name, &rgb[0],&rgb[1],&rgb[2],&k);
      if (num == 5)
      {
        shaders.set_and_activate(name, 
          new SpecularReflectiveShader(k * rgb));
      }
      else throw MakeException("Error");
      continue;
    }
    
    if (!strcmp(token,"glossy"))
    {
      double k, phong_exponent;
      Spectral3 kd_rgb, ks_rgb;
      std::unique_ptr<Texture> diffuse_texture, glossy_texture;
      char name[LINESIZE];
      
      int num = std::sscanf(line.c_str(),"glossy %s %lg %lg %lg %lg %lg\n",name,&ks_rgb[0], &ks_rgb[1], &ks_rgb[2], &k, &phong_exponent);
      if(num == 6)
      {
        while (true)
        {
          char buffer [LINESIZE];
          if (startswith(peek_line, "glossytexture"))
          {
            NextLine();
            num = std::sscanf(line.c_str(), "glossytexture %s\n", buffer);
            if (num == 1)
            {
              auto path = MakeFullPath(buffer);
              glossy_texture = std::make_unique<Texture>(path);
            }
            else throw MakeException("Error");
          }
          else break;
        }
        shaders.set_and_activate(name, new MicrofacetShader(
          k*ks_rgb, std::move(glossy_texture), phong_exponent
        ));
      }
      else throw MakeException("Error");
      continue;
    }
    
    
    if (!strcmp(token, "medium"))
    {
      Spectral3 sigma_s{0.}, sigma_a{1.};
      char name[LINESIZE];
      int num = std::sscanf(line.c_str(), "medium %s %lg %lg %lg %lg %lg %lg\n", name, &sigma_s[0], &sigma_s[1], &sigma_s[2], &sigma_a[0], &sigma_a[1], &sigma_a[2]);
      if(num == 1)
      {
        mediums.activate(name);
      }
      else if(num == 7)
      {
        auto *medium = new HomogeneousMedium(sigma_s, sigma_a, mediums.size());
        mediums.set_and_activate(
          name, medium);
      }
      else throw MakeException("Error");
      continue;
    }
  

    if (!strcmp(token, "pf"))
    {
      if (auto* medium = dynamic_cast<HomogeneousMedium*>(mediums()))
      {
        double g;
        char name[LINESIZE];
        int num = std::sscanf(line.c_str(),"pf %s %lg\n",name, &g);
        if (num > 0)
        {
          if (!strcmp(name, "rayleigh"))
          {
            medium->phasefunction.reset(new PhaseFunctions::Rayleigh());
          }
          else if (!strcmp(name, "henleygreenstein") && num>1)
          {
            medium->phasefunction.reset(new PhaseFunctions::HenleyGreenstein(g));
          }
          else
          {
            throw MakeException("Error");
          }
        }
        else
        {
          medium->phasefunction.reset(new PhaseFunctions::Uniform());
        }
      }
      else
      {
        throw MakeException("Warning: Phasefunction definition only admissible following a HomogeneousMedium definition.");
      }
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
        mediums.activate(name);
      }
      else if(num == 5)
      {
        auto medium = Atmosphere::MakeSimple(planet_center, radius, mediums.size());
        mediums.set_and_activate(
          name, medium.release());
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
        mediums.activate(name);
      }
      else if(num == 6)
      {
        auto medium = Atmosphere::MakeTabulated(planet_center, radius, datafile, mediums.size());
        mediums.set_and_activate(
          name, medium.release());
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
        scene->AddLight(std::make_unique<Sun>(total_power, dir_out, opening_angle));
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
      Spectral3 col;
      int num = sscanf(line.c_str(),"lddirection %lg %lg %lg %lg %lg %lg",
          &dir_out[0],&dir_out[1],&dir_out[2],
          &col[0],&col[1],&col[2]);
      Normalize(dir_out);
      if (num == 6)
      {
        scene->AddLight(std::make_unique<DistantDirectionalLight>(col, dir_out));
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
      Spectral3 col;
      int num = sscanf(line.c_str(),"lddome %lg %lg %lg %lg %lg %lg",
          &dir_up[0],&dir_up[1],&dir_up[2],
          &col[0],&col[1],&col[2]);
      Normalize(dir_up);
      if (num == 6)
      {
        scene->AddLight(std::make_unique<DistantDomeLight>(col, dir_up));
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
    Spectral3 col;
		int num = sscanf(line.c_str(),"l %lg %lg %lg %lg %lg %lg",
			   &pos[0],&pos[1],&pos[2],
			   &col[0],&col[1],&col[2]);
    col *= 1.0/255.9999;
		if (num == 3) {
			// light source with position only
			col = Spectral3(1,1,1);
			scene->AddLight(std::make_unique<PointLight>(col,pos));	
		} else if (num == 6) {
			// light source with position and color
			scene->AddLight(std::make_unique<PointLight>(col,pos));	
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
      scene->ParseNFF(fullpath, render_params);
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
      ParseMesh(fullpath.c_str());
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
    aiMatrix4x4 local_to_world;
    aiNode* node;
  };
public:
  void Read(const char *filename, NFFParser* parser, Scene *scene)
  {
    std::printf("Reading Mesh: %s\n", filename);
    this->aiscene = aiImportFile(filename, 0);
    this->scene = scene;
    this->parser = parser;
    
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
      DealWithTheMaterialOf(mesh);
      ReadMesh(mesh, ndref);
    }
  }
  
  void DealWithTheMaterialOf(const aiMesh* mesh)
  {
    if (mesh->mMaterialIndex < aiscene->mNumMaterials)
    {
      SetCurrentShader(aiscene->mMaterials[mesh->mMaterialIndex]);
    }
  }
  
  void ReadMesh(const aiMesh* mesh, const NodeRef &ndref)
  {
    auto m = ndref.local_to_world;
    auto m_linear = aiMatrix3x3(m);
    bool hasuv = mesh->GetNumUVChannels()>0;
    bool hasnormals = mesh->HasNormals();

    std::vector<int> triangle_indices; triangle_indices.reserve(1024);
    for (unsigned int face_idx = 0; face_idx < mesh->mNumFaces; ++face_idx)
    {
      const aiFace* face = &mesh->mFaces[face_idx];
      for (int i=2; i<face->mNumIndices; i++)
      {
        triangle_indices.push_back(face->mIndices[0]);
        triangle_indices.push_back(face->mIndices[i-1]);
        triangle_indices.push_back(face->mIndices[i]);
      }
    }
    std::vector<Double3> vertices; vertices.reserve(1024);
    for (int i=0; i<mesh->mNumVertices; ++i)
    {
      vertices.push_back(
            parser->ApplyTransform(
              aiVector3_to_myvector(
                m*mesh->mVertices[i])));
    }

    std::vector<Double3> normals; normals.reserve(1024);
    if (hasnormals)
    {
      for (int i=0; i<mesh->mNumVertices; ++i)
      {
        normals.push_back(
              parser->ApplyTransformNormal(
                aiVector3_to_myvector(
                  m_linear*mesh->mNormals[i])));
      }
    }
    std::vector<Double3> uvs; uvs.reserve(1024);
    if (hasuv)
    {
      for (int i=0; i<mesh->mNumVertices; ++i)
      {
        uvs.push_back(
              aiVector3_to_myvector(
                mesh->mTextureCoords[0][i]));
      }
    }

    for (int tri_start = 0; tri_start < triangle_indices.size(); tri_start+=3)
    {
      int a = triangle_indices[tri_start];
      int b = triangle_indices[tri_start+1];
      int c = triangle_indices[tri_start+2];
      if (hasuv || hasnormals)
      {
        Double3 uv[3] = {Double3{0.}, Double3{0.}, Double3{0.}};
        Double3 no[3];
        if (hasnormals)
        {
          no[0] = normals[a];
          no[1] = normals[b];
          no[2] = normals[c];
        }
        else
        {
          no[0] = no[1] = no[2] = Normalized(Cross(vertices[b]-vertices[a], vertices[c]-vertices[a]));
        }
        if (hasuv)
        {
          uv[0] = uvs[a];
          uv[1] = uvs[b];
          uv[2] = uvs[c];
        }
        parser->AssignCurrentMaterialParams(
          scene->AddPrimitive<TexturedSmoothTriangle>(
            vertices[a], vertices[b], vertices[c],
            no[0], no[1], no[2],
            uv[0], uv[1], uv[2]
         ));
      }
      else
      {
        parser->AssignCurrentMaterialParams(
          scene->AddPrimitive<Triangle>(
              vertices[a], vertices[b], vertices[c]));
      }
    }
  }
  
  void SetCurrentShader(const aiMaterial *mat)
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
      parser->shaders.activate(name);
  }

private:
  const aiScene *aiscene = { nullptr };
  Scene* scene = { nullptr };
  NFFParser *parser = {nullptr};

  inline Double3 aiVector3_to_myvector(const aiVector3D &v)
  {
    return Double3{v[0], v[1], v[2]};
  }
};


void NFFParser::ParseMesh(const char* filename)
{
  AssimpReader().Read(filename, this, scene);
}


void Scene::ParseNFF(const std::string &filename, RenderingParameters *render_params)
{
  std::ifstream is(filename);
  if (!is.good())
  {
    throw std::runtime_error(strconcat("Could not open input file", filename));
  }
  NFFParser(this, render_params, is, filename).Parse();
}


void Scene::ParseNFFString(const std::string &scenestr, RenderingParameters *render_params)
{
  std::istringstream is(scenestr);
  NFFParser(this, render_params, is, std::string()).Parse();
}

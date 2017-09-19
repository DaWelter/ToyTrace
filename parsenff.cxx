#include "scene.hxx"
#include "perspectivecamera.hxx"
#include "sphere.hxx"
#include "infiniteplane.hxx"
#include "triangle.hxx"
#include "shader.hxx"

#define LINESIZE 1000

#include <vector>
#include <cstdio>
#include <unordered_map>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


class NFFParser
{
  Shader *currentShader;
  Scene* scene;
  std::unordered_map<std::string, Shader*> shaders;
  friend class AssimpReader;
public:
  NFFParser(Scene* _scene) :
    scene(_scene)
  {
    // just to have a default shader, in case the file doesn't define one !
    currentShader = new DiffuseShader(Double3(0.8, 0.8, 0.8)); // EyeLightShader(Double3(1,1,1));
  }
  void Parse(char *fileName);
private:
  
  void ParseMesh(char *filename);
  
  void SetCurrentShader(const std::string &name)
  {
    auto shader_iter = shaders.find(name);
    if (shader_iter != shaders.end())
    {
      currentShader = shader_iter->second;
    }
    else
    {
      std::printf("Error: Material %s not defined. Define it in the NFF file prior to referencing it.\n", name.c_str());
      exit(0);
    }
  }
  
  void SetCurrentShader(const char* name, Shader* shader)
  {
    currentShader = shader;
    shaders[name] = shader;
  }
};



void NFFParser::Parse(char *fileName)
{
  char line[LINESIZE+1];
  char token[LINESIZE+1];
  char *str;
  int i;

  FILE *file = fopen(fileName,"r");
  if (!file) 
  {
    std::cerr << "could not open input file " << fileName << std::endl;
    exit(1);
  }
  
  int line_no = 0;
  while (!feof(file)) 
  {
    str = std::fgets(line,LINESIZE,file);
    if (str == nullptr)
      continue;
    //std::cout << (++line_no) << ": " << str;
    
    if (str[0] == '#') // '#' : comment
      continue;
    
    int numtokens = sscanf(line,"%s",token);
    if (numtokens <= 0) // empty line, except for whitespaces
      continue; 
    
    /* start new group */
    
    if (!strcmp(token,"begin_hierarchy")) {
      line[strlen(line)-1] = 0; // remove trailing eol indicator '\n'
      //Parse(file, fileName);
      continue;
    }

    /* end group */

    if (!strcmp(token,"end_hierarchy")) {
      //return;
      continue;
    }
    
    /* camera */

    if (!strcmp(token,"v")) {
      // FORMAT:
      //     v
      //     from %lg %lg %lg
      //     at %lg %lg %lg
      //     up %lg %lg %lg
      //     angle %lg
      //     hither %lg
      //     resolution %d %d
      Double3 pos, at, up;
      double angle, hither;
      int resX, resY;
      fscanf(file,"from %lg %lg %lg\n",&pos[0],&pos[1],&pos[2]);
      fscanf(file,"at %lg %lg %lg\n",&at[0],&at[1],&at[2]);
      fscanf(file,"up %lg %lg %lg\n",&up[0],&up[1],&up[2]);
      fscanf(file,"angle %lg\n",&angle);
      fscanf(file,"hither %lg\n",&hither);
      fscanf(file,"resolution %d %d\n",&resX,&resY);
      scene->SetCamera<PerspectiveCamera>(pos,at-pos,up,angle,resX,resY);
      
      continue;
    }
    
    /* sphere */

    if (!strcmp(token,"s")) {
      Double3 pos;
      double rad;
      sscanf(str,"s %lg %lg %lg %lg",&pos[0],&pos[1],&pos[2],&rad);
      scene->AddPrimitive(std::make_unique<Sphere>(pos,rad,currentShader));
      continue;
    }

    /* polygon (with normals) */
    
    if (!strcmp(token,"pp")) {
      int vertices;
      sscanf(str,"pp %d",&vertices);
      Double3 *vertex = new Double3[vertices];
      Double3 *normal = new Double3[vertices];

      for (i=0;i<vertices;i++) {
			fscanf(file,"%lg %lg %lg %lg %lg %lg\n",
			&vertex[i][0],&vertex[i][1],&vertex[i][2],
			&normal[i][0],&normal[i][1],&normal[i][2]);
      }
      assert(currentShader != NULL);
      for (i=2;i<vertices;i++) {
		scene->AddPrimitive
			(std::make_unique<SmoothTriangle>(
				vertex[0],
				vertex[i-1],
				vertex[i],
				normal[i-1],
				normal[i],
				normal[0],
				currentShader));
      }
      delete[] vertex;
      delete[] normal;
      continue;
    }

/* polygon (with normals and uv) */
    if (!strcmp(token,"tpp")) 
	{
		int vertices;
		sscanf(str,"tpp %d",&vertices);
		Double3 *vertex = new Double3[vertices];
		Double3 *normal = new Double3[vertices];
		Double3 *uv     = new Double3[vertices];

		for (i=0;i<vertices;i++) {
			fscanf(file,"%lg %lg %lg %lg %lg %lg %lg %lg %lg\n",
			&vertex[i][0],&vertex[i][1],&vertex[i][2],
			&normal[i][0],&normal[i][1],&normal[i][2],
			&uv[i][0],&uv[i][1],&uv[i][2]);
		}
		assert(currentShader != NULL);
		for (i=2;i<vertices;i++) {
		scene->AddPrimitive(
			std::make_unique<TexturedSmoothTriangle>(
				vertex[0],
				vertex[i-1],
				vertex[i],
				normal[i-1],
				normal[i],
				normal[0],
				uv[i-1],
				uv[i],
				uv[0],
				currentShader));
		}
		delete[] vertex;
		delete[] normal;
		continue;
    }

    /* background color */

    if (!strcmp(token,"b")) {
      sscanf(line,"b %lg %lg %lg",&scene->bgColor[0],&scene->bgColor[1],&scene->bgColor[2]);
      scene->bgColor /= 255;
      continue;
    }
    
    /* polygon */

    if (!strcmp(token,"p")) {
		int vertices;
		sscanf(str,"p %d",&vertices);
		Double3 *vertex = new Double3[vertices];
		for (i=0;i<vertices;i++) {
			fscanf(file,"%lg %lg %lg\n",&vertex[i][0],&vertex[i][1],&vertex[i][2]);
		}
		assert(currentShader != NULL);
		for (i=2;i<vertices;i++) {
			scene->AddPrimitive
				(std::make_unique<Triangle>(vertex[0],
					vertex[i-1],
					vertex[i],
					currentShader));
			}
		delete[] vertex;
		continue;
    }
    
    /* include new NFF file */
    
    if (!strcmp(token,"include")) {
      if (!fgets(line,LINESIZE,file)) {
			std::cerr << " error in include, cannot read filename to include" << std::endl;
			exit(0);
      }
      line[strlen(line)-1] = 0; // remove trailing eol indicator '\n'
      std::cout << "including file " << line << std::endl;
      Parse(line);
      continue;
    }
    
    /* shader parameters */
    
    if (!strcmp(token,"f")) {
      double r,g,b,kd,ks,shine,t,ior;
      char texture[LINESIZE];
      char name[LINESIZE];
      
      int num = sscanf(line,"f %s %lg %lg %lg %lg %lg %lg %lg %lg %s\n",name, &r,&g,&b,&kd,&ks,&shine,&t,&ior,texture);
      Double3 color(r,g,b);
      if(num==9) 
      {
        //currentShader = new PhongShader(color,color,Double3(1.),0.1,kd,ks,shine,ks);
        SetCurrentShader(name, 
                         new DiffuseShader(color)
                        );
      } 
      else if(num==10) 
      {
        SetCurrentShader(name, 
                         new DiffuseShader(color)
                        );
      }
      else {
        std::cout << "error in " << fileName << " : " << line << std::endl;
      }
      this->shaders[name] = currentShader;
      continue;
    }
    
    /* lightsource */
    
    if (!strcmp(token,"l")) 
	{
		Double3 pos;
    Spectral col;
		int num = sscanf(line,"l %lg %lg %lg %lg %lg %lg",
			   &pos[0],&pos[1],&pos[2],
			   &col[0],&col[1],&col[2]);
    col *= 1.0/255.9999;
		if (num == 3) {
			// light source with position only
			col = Spectral(1,1,1);
			scene->AddLight(std::make_unique<PointLight>(col,pos));	
		} else if (num == 6) {
			// light source with position and color
			scene->AddLight(std::make_unique<PointLight>(col,pos));	
		} else {
			std::cout << "error in " << fileName << " : " << line << std::endl;
		}
		continue;
    }

	
// 	if(!strcmp(token,"sl"))
// 	{
// 		Double3 pos,dir,col;
// 		double min=0,max=10;
// 		int num = sscanf(line,"sl %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg",
// 				&pos[0],&pos[1],&pos[2],&dir[0],&dir[1],&dir[2],&col[0],&col[1],&col[2],&min,&max); 
// 		if(num == 11) {
// 			scene->AddLight(std::make_unique<SpotLight>(col,pos,dir,min,max));
// 		}
// 		else {
// 			std::cout << "error in "<<fileName<<" : " << line << std::endl;
// 		}
// 		continue;
// 	}
	
	
	if (!strcmp(token, "m"))
  {
    char meshfile[LINESIZE];
    int num = sscanf(line, "m %s", meshfile);
    if (num == 1)
    {
      ParseMesh(meshfile);
    }
    else
    {
      std::cout << "error in " << fileName << " : " << line << std::endl;
    }
    continue;
  }
  
  
  /* unknown command */
    
    std::cout << "error in " << fileName << " : " << line << std::endl;
    exit(0);
  }
  
  fclose(file);
};


// Example see: https://github.com/assimp/assimp/blob/master/samples/SimpleOpenGL/Sample_SimpleOpenGL.c
class AssimpReader
{
public:
  void Read(char *filename, NFFParser* parser, Scene *scene)
  {
    std::printf("Reading Mesh: %s\n", filename);
    this->aiscene = aiImportFile(filename, 0);
    this->scene = scene;
    this->parser = parser;
    
    if (!aiscene)
    {
      std::printf("Error: could not load file.");
      exit(0);
    }
    
    std::vector<aiNode*> nodestack{ aiscene->mRootNode };
    while (!nodestack.empty())
    {
      const auto *nd = nodestack.back();
      nodestack.pop_back();
      for (unsigned int i = 0; i< nd->mNumChildren; ++i)
      {
        nodestack.push_back(nd->mChildren[i]);
      }
      
      ReadNode(nd);
    }
  }
  
private:
  void ReadNode(const aiNode* nd)
  {
    for (unsigned int mesh_idx = 0; mesh_idx < nd->mNumMeshes; ++mesh_idx)
    {
      const aiMesh* mesh = aiscene->mMeshes[nd->mMeshes[mesh_idx]];
      std::printf("Mesh %i (%s), mat_idx=%i\n", mesh_idx, mesh->mName.C_Str(), mesh->mMaterialIndex);
      DealWithTheMaterialOf(mesh);
      ReadMesh(mesh);
    }
  }
  
  void DealWithTheMaterialOf(const aiMesh* mesh)
  {
    if (mesh->mMaterialIndex >= 0 && mesh->mMaterialIndex < aiscene->mNumMaterials)
    {
      SetCurrentShader(aiscene->mMaterials[mesh->mMaterialIndex]);
    }
  }
  
  void ReadMesh(const aiMesh* mesh)
  {
    for (unsigned int face_idx = 0; face_idx < mesh->mNumFaces; ++face_idx)
    {
      const aiFace* face = &mesh->mFaces[face_idx];
      auto vertex0 = aiVector3_to_myvector(mesh->mVertices[face->mIndices[0]]);
      for (int i=2; i<face->mNumIndices; i++)
      {
        auto vertex1 = aiVector3_to_myvector(mesh->mVertices[face->mIndices[i-1]]);
        auto vertex2 = aiVector3_to_myvector(mesh->mVertices[face->mIndices[i  ]]);
        scene->AddPrimitive(std::make_unique<Triangle>(
            vertex0,
            vertex1,
            vertex2,
            parser->currentShader));
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
    
    parser->SetCurrentShader(name);
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


void NFFParser::ParseMesh(char *filename)
{
  AssimpReader().Read(filename, this, scene);
}


void Scene::ParseNFF(char *fileName)
{
  // parse file, add all items to 'primitives'
  NFFParser(this).Parse(fileName);
}




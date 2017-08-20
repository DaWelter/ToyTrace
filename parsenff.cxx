#include "scene.hxx"
#include "perspectivecamera.hxx"
#include "sphere.hxx"
#include "infiniteplane.hxx"
#include "triangle.hxx"
#include "shader.hxx"

#define LINESIZE 1000

void Scene::ParseNFF(char *fileName)
{
  // parse file, add all items to 'primitives'
  ParseNFF(NULL,fileName,&primitives);
}

void Scene::ParseNFF(FILE *fileToUse, char *fileName, Group *groupToAddTo)
{
  char line[LINESIZE+1];
  char token[LINESIZE+1];
  char *str;
  int i;
  /* open file */

  FILE *file = fileToUse;
  if (!file) {
    file = fopen(fileName,"r");
    if (!file) {
      std::cerr << "could not open input file " << fileName << std::endl;
      exit(1);
    }
  }
  
  // just to have a default shader, in case the file doesn't define one !
  static Shader *currentShader = new EyeLightShader(Double3(1,1,1));
  
  /* parse lines */
  
  while ((str = fgets(line,LINESIZE,file)) && !feof(file)) {
    if (str[0] == '#') // '#' : comment
      continue;
    
    int numtokens = sscanf(line,"%s",token);
    if (numtokens <= 0) // empty line, except for whitespaces
      continue; 
    
    /* start new group */
    
    if (!strcmp(token,"begin_hierarchy")) {
      line[strlen(line)-1] = 0; // remove trailing eol indicator '\n'
      Group *subGroup = new Group;
      groupToAddTo->Add(subGroup);
      ParseNFF(file,fileName,subGroup);
      continue;
    }

    /* end group */

    if (!strcmp(token,"end_hierarchy")) {
      return;
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

      if (camera)
	delete camera;
      camera = new PerspectiveCamera(pos,at-pos,up,angle,resX,resY);
      
      continue;
    }
    
    /* sphere */

    if (!strcmp(token,"s")) {
      Double3 pos;
      double rad;
      sscanf(str,"s %lg %lg %lg %lg",&pos[0],&pos[1],&pos[2],&rad);
      groupToAddTo->Add(new Sphere(pos,rad,currentShader));
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
		groupToAddTo->Add
			(new SmoothTriangle(
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
		groupToAddTo->Add
			(new TexturedSmoothTriangle(
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
      sscanf(line,"b %lg %lg %lg",&bgColor[0],&bgColor[1],&bgColor[2]);
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
			groupToAddTo->Add
				(new Triangle(vertex[0],
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
      
      Group *subGroup = new Group;
      groupToAddTo->Add(subGroup);
      ParseNFF(NULL,line,subGroup);
      continue;
    }
    
    /* shader parameters */
    
    if (!strcmp(token,"f")) {
      double r,g,b,kd,ks,shine,t,ior;
	  char texture[LINESIZE];
      
	  int num = sscanf(line,"f %lg %lg %lg %lg %lg %lg %lg %lg %s\n",&r,&g,&b,&kd,&ks,&shine,&t,&ior,texture);
      Double3 color(r,g,b);
	  
	  if(num==8) {
		  //currentShader = new PhongShader(color,color,Double3(1.),0.1,kd,ks,shine,ks);
		  currentShader = new ReflectiveEyeLightShader(color,ks);
	  } else if(num==9) {
		  currentShader = new TexturedEyeLightShader(color*kd,texture);
		}
	  else {
		  std::cout << "error in " << fileName << " : " << line << std::endl;
	  }
      continue;
    }
    
    /* lightsource */
    
    if (!strcmp(token,"l")) 
	{
		Double3 pos, col;
		int num = sscanf(line,"l %lg %lg %lg %lg %lg %lg",
			   &pos[0],&pos[1],&pos[2],
			   &col[0],&col[1],&col[2]);

		if (num == 3) {
			// light source with position only
			col = Double3(1,1,1);
			AddLight(new PointLight(col,pos));	
		} else if (num == 6) {
			// light source with position and color
			AddLight(new PointLight(col,pos));	
		} else {
			std::cout << "error in " << fileName << " : " << line << std::endl;
		}
		continue;
    }

	
	if(!strcmp(token,"sl"))
	{
		Double3 pos,dir,col;
		double min=0,max=10;
		int num = sscanf(line,"sl %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg %lg",
				&pos[0],&pos[1],&pos[2],&dir[0],&dir[1],&dir[2],&col[0],&col[1],&col[2],&min,&max); 
		if(num == 11) {
			AddLight(new SpotLight(col,pos,dir,min,max));
		}
		else {
			std::cout << "error in "<<fileName<<" : " << line << std::endl;
		}
		continue;
	}

    /* unknown command */
    
    std::cout << "error in " << fileName << " : " << line << std::endl;
    exit(0);
  }
  
  if (fileToUse)
    fclose(file);
};

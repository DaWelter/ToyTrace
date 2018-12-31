#pragma once

#include "very_strong_typedef.hxx"


struct tag_MaterialIndex {};
struct tag_ShaderIndex {};
struct tag_MediumIndex {};
using MaterialIndex = very_strong_typedef<short, tag_MaterialIndex>;
using ShaderIndex = very_strong_typedef<short, tag_ShaderIndex>;
using MediumIndex = very_strong_typedef<short, tag_MaterialIndex>;

class Shader;
class Medium;
class Scene;
class Shader;
class Medium;
class Camera;
class Sampler;
class Geometry;
class Texture;

namespace RadianceOrImportance {
class AreaEmitter;
class PointEmitter;
class EnvironmentalRadianceField;
}

struct SurfaceInteraction;
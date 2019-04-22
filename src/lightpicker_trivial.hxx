#pragma once

#include "scene.hxx"
#include "shader.hxx"
#include "light.hxx"
#include "util.hxx"

class LightRef
{
    std::uint32_t type : 2;
    std::uint32_t geom_idx : 30;
    std::uint32_t prim_idx;
    friend class LightPicker;
};
static_assert(sizeof(LightRef) == 8);


class LightPicker
{
public:
    virtual LightRef PickLight(Sampler &sampler) const;
    virtual void ObserveLightContribution(const LightRef &lr, const Spectral3 &radiance);
    
    template<class Visitor> 
    void PickLight(Sampler &sampler, Visitor &&visitor) const
    {
        LightRef lr = PickLight(sampler);
        switch (lr.type)
        {
            // case ...
            
        }
    }
};


using  AlgorithmParameters = RenderingParameters;
namespace ROI = RadianceOrImportance;

/* Selects lights randomly with uniform probability distributions. */
class TrivialLightPicker
{
  const Scene &scene;
  ToyVector<std::pair<int,int>> arealight_refs;
  ToyVector<int> volume_light_refs;
  
  void FindAreaLightGeometry();
  void FindVolumeLightGeometry();
  
  static constexpr int IDX_PROB_ENV = 0;
  static constexpr int IDX_PROB_AREA = 1;
  static constexpr int IDX_PROB_POINT = 2;
  static constexpr int IDX_PROB_VOLUME = 3;
  std::array<double, 4> emitter_type_selection_probabilities;
  
public:
  TrivialLightPicker(const Scene &_scene);
  
  // Invokes visitor with the overload pertaining to the selected light type.
  template<class Visitor> 
  void PickLight(Sampler &sampler, Visitor &&visitor)
  {
    int which_kind = TowerSampling<4>(emitter_type_selection_probabilities.data(), sampler.Uniform01());
    switch (which_kind)
    {
      case IDX_PROB_ENV:
      {
        double prob = emitter_type_selection_probabilities[which_kind];
        visitor(scene.GetTotalEnvLight(), prob);
      }
      break;
      case IDX_PROB_POINT:
      {
        const int n = scene.GetNumLights();
        int idx = sampler.UniformInt(0, n-1);
        double prob = emitter_type_selection_probabilities[which_kind]/n;
        visitor(scene.GetLight(idx), prob);
      }
      break;
      case IDX_PROB_AREA:
      {
        const int n = arealight_refs.size();
        auto [geom_idx, prim_idx] = arealight_refs[sampler.UniformInt(0, n-1)];
        double prob = emitter_type_selection_probabilities[which_kind]/n;
        visitor(PrimRef{&scene.GetGeometry(geom_idx),prim_idx}, prob);
      }
      break;
      case IDX_PROB_VOLUME:
      {
        const int n = volume_light_refs.size();
        auto idx = volume_light_refs[sampler.UniformInt(0, n-1)];
        double prob = emitter_type_selection_probabilities[which_kind]/n;
        return visitor(*ASSERT_NOT_NULL(scene.GetMaterial(idx).medium), prob);
      }
      break;
    }
  }
  
  double PmfOfLight(const ROI::EnvironmentalRadianceField &) const
  {
    return emitter_type_selection_probabilities[IDX_PROB_ENV];
  }

  double PmfOfLight(const ROI::PointEmitter &) const
  {
    return emitter_type_selection_probabilities[IDX_PROB_POINT]/scene.GetNumLights();
  }

  double PmfOfLight(const ROI::AreaEmitter &) const
  {
    return emitter_type_selection_probabilities[IDX_PROB_AREA]/arealight_refs.size();
  }
  
  double PmfOfLight(const Medium &) const
  {
    return emitter_type_selection_probabilities[IDX_PROB_VOLUME]/volume_light_refs.size();
  }
};


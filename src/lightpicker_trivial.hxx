#pragma once

#include "scene.hxx"
#include "shader.hxx"
#include "light.hxx"
#include "util.hxx"


class LightPickerCommon
{
protected:
  const Scene &scene;
  ToyVector<std::pair<int, int>> arealight_refs;
  ToyVector<int> volume_light_refs;
  
  inline static constexpr int IDX_PROB_ENV = 0;
  inline static constexpr int IDX_PROB_AREA = 1;
  inline static constexpr int IDX_PROB_POINT = 2;
  inline static constexpr int IDX_PROB_VOLUME = 3;
  inline static constexpr int NUM_LIGHT_TYPES = 4;

  LightPickerCommon(const Scene &scene);
  void FindAreaLightGeometry();
  void FindVolumeLightGeometry();
};


using  AlgorithmParameters = RenderingParameters;
namespace ROI = RadianceOrImportance;

/* Selects lights randomly with uniform probability distributions. */
class TrivialLightPicker : public LightPickerCommon
{
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
        const int n = scene.GetNumPointLights();
        int idx = sampler.UniformInt(0, n-1);
        double prob = emitter_type_selection_probabilities[which_kind]/n;
        visitor(scene.GetPointLight(idx), prob);
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
    return emitter_type_selection_probabilities[IDX_PROB_POINT]/scene.GetNumPointLights();
  }

  double PmfOfLight(const PrimRef &) const
  {
    return emitter_type_selection_probabilities[IDX_PROB_AREA]/arealight_refs.size();
  }
  
  double PmfOfLight(const Medium &) const
  {
    return emitter_type_selection_probabilities[IDX_PROB_VOLUME]/volume_light_refs.size();
  }
};



#if 1
namespace UcbLightPickerImpl
{

class LightRef
{
  std::uint32_t type : 2;
  std::uint32_t geom_idx : 30;
  std::uint32_t prim_idx;
  friend class LightPicker;
};
static_assert(sizeof(LightRef) == 8);

class Stats
{
public:
  Stats();
  Stats(int num_arms);
  void ShareWith(Stats &other);
  void ObserveReturn(int arm, double value);
  //static void Update(Span<Stats*> stats);
  void UpdateAllShared(); // Call this on one of the element in the group. Will combine statistics and update all of them.
  Span<const double> GetCummulativeProbs();
private:
  // Per Thread
  Eigen::ArrayXd mean;
  Eigen::ArrayXd sqr_sum;
  Eigen::ArrayXi count;
  // Shared for all threads
  // For convencience and speed(?)
  Eigen::Map<Eigen::ArrayXd> cummulative_probs;
  std::shared_ptr<Eigen::ArrayXd> shared_cummulative_probs;
  int step_count = 0;
  Stats* next = nullptr; // Circular Link list
};


class UcbLightPicker : public LightPickerCommon
{
public:
  class PickCallbacks
  {
  public:
    virtual void operator()(const ROI::EnvironmentalRadianceField &env, double prob) = 0;
    virtual void operator()(const ROI::PointEmitter &light, double prob) = 0;
    virtual void operator()(const PrimRef& prim_ref, double prob) = 0;
    virtual void operator()(const Medium &medium, double prob) = 0;
  };

  UcbLightPicker(const Scene &scene);
  void ShareStatsWith(UcbLightPicker &other);
  void PickLight(Sampler &sampler, PickCallbacks &cb) const;
  void ObserveLightContribution(const LightRef &lr, const Spectral3 &radiance);
  //static void Update(Span<UcbLightPicker*> pickers);
  void UpdateAllShared();
  double PmfOfLight(const ROI::EnvironmentalRadianceField &) const;
  double PmfOfLight(const ROI::PointEmitter &) const;
  double PmfOfLight(const PrimRef &) const;
  double PmfOfLight(const Medium &) const;

private:
  Stats light_sampler[NUM_LIGHT_TYPES];
  std::array<double, NUM_LIGHT_TYPES> emitter_type_selection_probabilities;
};

}

using UcbLightPickerImpl::UcbLightPicker;
#endif

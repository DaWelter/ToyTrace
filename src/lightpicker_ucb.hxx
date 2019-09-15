#pragma once

#include "lightpicker_trivial.hxx"
#include "linklist.hxx"


namespace UcbLightPickerImpl
{

struct LightRef
{
  std::uint32_t type : 2;
  std::uint32_t idx : 30;
};
static_assert(sizeof(LightRef) == 4);

// One of these class instances per worker.
// Accumulate statistics about radiance received from each light.
class Stats : public LinkListBase<Stats>
{
public:
  Stats();
  void Init(int arm_count);
  void InitSharedWith(Stats &other);
  void ObserveReturn(int arm, double value);
  void UpdateAllShared(double &out_weight_sum); // Call this on one of the element in the group. Will combine statistics and update all of them.
  int SampleArm(double r) const;
  double ArmProbability(int i) const;
  int ArmCount() const;
  void Print(std::ostream &os) const;
private:
  // Per Thread
  OnlineVariance::ArrayAccumulator<double> accum;
  // Shared for all threads
  // For convencience and speed(?)
  Span<double> cummulative_probs;
  // Where the memory is owned.
  std::shared_ptr<Eigen::ArrayXd> shared_cummulative_probs;
  int step_count = 0;
};


class UcbLightPicker : public LightPickerCommon, public LinkListBase<UcbLightPicker>
{
public:
  // class PickCallbacks
  // {
  // public:
  //   virtual void operator()(const ROI::EnvironmentalRadianceField &env, double prob) = 0;
  //   virtual void operator()(const ROI::PointEmitter &light, double prob) = 0;
  //   virtual void operator()(const PrimRef& prim_ref, double prob) = 0;
  //   virtual void operator()(const Medium &medium, double prob) = 0;
  // };
  UcbLightPicker(const Scene &scene);
  void Init();
  void InitSharedWith(UcbLightPicker &other);
  
  void ObserveLightContribution(const LightRef &lr, const Spectral3 &radiance);
  void ObserveLightContribution(const LightRef &lr, float value);
  void UpdateAllShared();

  float RadianceToObservationValue(const Spectral3 &r);

  template<class Visitor>
  void PickLight(Sampler &sampler, Visitor &&visitor);
  double PmfOfLight(const ROI::EnvironmentalRadianceField &) const;
  double PmfOfLight(const ROI::PointEmitter &) const;
  double PmfOfLight(const PrimRef &) const;
  double PmfOfLight(const Medium &) const;

  void Print(std::ostream &os) const;
private:
  Stats light_sampler[NUM_LIGHT_TYPES];
  Eigen::Array<double, 4, 1> light_type_selection_probs;
};


//======================================================================
//======================================================================
//======================================================================


inline int Stats::ArmCount() const
{
  return accum.Size();
}


inline void Stats::ObserveReturn(int arm, double value)
{
  accum.Add(arm, value);
}


inline int Stats::SampleArm(double r) const
{
  return TowerSamplingBisection<double>(cummulative_probs, r);
}


inline double Stats::ArmProbability(int i) const
{
  return TowerSamplingProbabilityFromCmf<double>(cummulative_probs, i);
}


//======================================================================
//======================================================================
//======================================================================


inline double UcbLightPicker::PmfOfLight(const ROI::EnvironmentalRadianceField &) const
{
  return light_type_selection_probs[IDX_PROB_ENV];
  //return light_type_sampler.ArmProbability(IDX_PROB_ENV);
}

inline double UcbLightPicker::PmfOfLight(const ROI::PointEmitter &e) const
{
  return light_type_selection_probs[IDX_PROB_POINT]*
    light_sampler[IDX_PROB_POINT].ArmProbability(e.scene_index);
}

inline double UcbLightPicker::PmfOfLight(const PrimRef &ref) const
{
  auto index = scene.GetAreaLightIndex(ref);
  return light_type_selection_probs[IDX_PROB_AREA]*
    light_sampler[IDX_PROB_AREA].ArmProbability(index);
}

inline double UcbLightPicker::PmfOfLight(const Medium &) const
{
  assert(!"implemented");
  return 0;
}


template<class Visitor>
inline void UcbLightPicker::PickLight(Sampler &sampler, Visitor &&visitor)
{
  const int which_kind = TowerSampling<NUM_LIGHT_TYPES>(
    light_type_selection_probs.data(), sampler.Uniform01());
  const double kind_selection_prob = light_type_selection_probs[which_kind];
  const int idx = light_sampler[which_kind].SampleArm(sampler.Uniform01());
  const double item_selection_prob = light_sampler[which_kind].ArmProbability(idx); 
  const double prob = kind_selection_prob * item_selection_prob;
  const LightRef ref{ (uint32_t)which_kind, (uint32_t)idx };
  switch (which_kind)
  {
  case IDX_PROB_ENV:
  {
    visitor(scene.GetTotalEnvLight(), prob, ref);
  }
  break;
  case IDX_PROB_POINT:
  {
    
    visitor(scene.GetPointLight(idx), prob, ref);
  }
  break;
  case IDX_PROB_AREA:
  {
    auto prim_ref = scene.GetPrimitiveFromAreaLightIndex(idx);
    visitor(prim_ref, prob, ref);
  }
  break;
  case IDX_PROB_VOLUME:
  {
    assert(!"not implemented");
    //const int n = isize(volume_light_refs);
    //auto idx = volume_light_refs[sampler.UniformInt(0, n-1)];
    //double prob = emitter_type_selection_probabilities[which_kind]/n;
    //return visitor(*ASSERT_NOT_NULL(scene.GetMaterial(idx).medium), prob);
  }
  break;
  }
}


inline float UcbLightPickerImpl::UcbLightPicker::RadianceToObservationValue(const Spectral3 & r)
{
  return static_cast<float>(r.sum());
}


}

using UcbLightPickerImpl::UcbLightPicker;
using UcbLightPickerImpl::LightRef;

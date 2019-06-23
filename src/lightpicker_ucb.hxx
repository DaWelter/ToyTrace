#pragma once

#include "lightpicker_trivial.hxx"
#include "linklist.hxx"

#if 1
namespace UcbLightPickerImpl
{

struct LightRef
{
  std::uint32_t type : 2;
  std::uint32_t geom_idx : 30;
  std::uint32_t prim_idx;
};
static_assert(sizeof(LightRef) == 8);

// One of these class instances per worker.
// Accumulate statistics about radiance received from each light.
class Stats : public LinkListBase<Stats>
{
public:
  Stats();
  Stats(int num_arms);
  void ShareWith(Stats &other);
  void ObserveReturn(int arm, double value);
  void UpdateAllShared(); // Call this on one of the element in the group. Will combine statistics and update all of them.
  Span<const double> GetCummulativeProbs();
private:
  // Per Thread
  Eigen::ArrayXd mean;
  Eigen::ArrayXd sqr_sum;
  Eigen::ArrayXi count;
  // Shared for all threads
  // For convencience and speed(?)
  Span<float>  cummulative_probs;
  // Where the memory is owned.
  std::shared_ptr<Eigen::ArrayXd> shared_cummulative_probs;
  int step_count = 0;
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
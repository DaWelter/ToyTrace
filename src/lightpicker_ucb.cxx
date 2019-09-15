#include "lightpicker_ucb.hxx"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <numeric>

#include "gtest/gtest.h"


namespace UcbLightPickerImpl
{

Stats::Stats()
  : LinkListBase<Stats>(),
    accum(0)
{
}


void Stats::Init(int arm_count)
{
  step_count = 0;
  accum = decltype(accum)(arm_count);
  shared_cummulative_probs = std::make_shared<Eigen::ArrayXd>(arm_count);
  cummulative_probs = AsSpan(*shared_cummulative_probs);
}


void Stats::InitSharedWith(Stats & other)
{
  step_count = 0;
  accum = decltype(accum)(other.ArmCount());

  shared_cummulative_probs = other.shared_cummulative_probs;
  cummulative_probs = AsSpan(*shared_cummulative_probs);
  CircularLinkList<Stats>::append(other, *this);
}


void Stats::UpdateAllShared(double &out_weight_sum)
{
  ++step_count;
  // To make it very likely that unsampled arms are visited.
  const double large_weight = std::numeric_limits<double>::max()*1.e-9;
  const double logt = std::log(step_count);

  const int arm_count = ArmCount();
  if (arm_count <= 0)
  {
    out_weight_sum = 0;
    return;
  }

  OnlineVariance::ArrayAccumulator<double> accum(arm_count);
  for (Stats &stats : CircularLinkList<Stats>::rangeStartingAt(*this))
  {
    accum.Add(stats.accum);
  }

  const auto &count = accum.Counts();
  const auto &mean = accum.Mean();
  const auto &var = accum.Var(/*fill_value=*/large_weight);

  for (int i = 0; i < arm_count; ++i)
  {
      if (count[i] >= 1)
      {
        auto confidence_bound = std::sqrt(2.* logt / count[i]);
        cummulative_probs[i] = mean[i] + var[i] + confidence_bound;
      }
      else
        cummulative_probs[i] = large_weight;
      assert(std::isfinite(cummulative_probs[i]));
  }
  TowerSamplingInplaceCumSum(cummulative_probs);
  out_weight_sum = cummulative_probs[arm_count - 1];
  TowerSamplingNormalize(cummulative_probs);
}


void UcbLightPickerImpl::Stats::Print(std::ostream & os) const
{
  for (int i = 0; i < ArmCount(); ++i)
  {
    os << strformat("p[%s]=%s\n", i, TowerSamplingProbabilityFromCmf(cummulative_probs, i));
  }
}



UcbLightPicker::UcbLightPicker(const Scene & scene)
  : LightPickerCommon{ scene },
    LinkListBase<UcbLightPicker>()
{
}

void UcbLightPicker::Init()
{
  light_sampler[LightPickerCommon::IDX_PROB_POINT].Init(scene.GetNumPointLights());
  light_sampler[LightPickerCommon::IDX_PROB_AREA].Init(scene.GetNumAreaLights());
  light_sampler[LightPickerCommon::IDX_PROB_ENV].Init(scene.HasEnvLight() ? 1 : 0);
  light_sampler[LightPickerCommon::IDX_PROB_VOLUME].Init(0);
  //int num_lights = std::transform_reduce(
  //    light_sampler,  light_sampler+NUM_LIGHT_TYPES, 0, 
  //    std::plus<>(),
  //    [](const Stats &s) { return s.ArmCount(); });
  for (int i = 0; i < NUM_LIGHT_TYPES; ++i)
    light_type_selection_probs[i] = static_cast<double>(light_sampler[i].ArmCount());
}

void UcbLightPicker::InitSharedWith(UcbLightPicker &other)
{
  assert(&other != this); // Probably not what I want.
  for (int i = 0; i < NUM_LIGHT_TYPES; ++i)
  {
    light_sampler[i].InitSharedWith(other.light_sampler[i]);
    light_type_selection_probs[i] = other.light_type_selection_probs[i];
  }
  CircularLinkList<UcbLightPicker>::append(other, *this);
}


void UcbLightPicker::ObserveLightContribution(const LightRef & lr, const Spectral3 & radiance)
{
  assert(lr.type >= 0 && lr.type < NUM_LIGHT_TYPES);
  // This should maybe be the integral over the full spectrum, to get the total power.
  // Instead I sort of take the average over the spectrum. It is not the same because
  // this way I'm factoring in the variance across the spectrum. Not sure if this is right ...
  for (int lambda = 0; lambda < static_size<Spectral3>(); ++lambda)
  {
    light_sampler[lr.type].ObserveReturn(lr.idx, radiance[lambda]);
  }
}


void UcbLightPickerImpl::UcbLightPicker::ObserveLightContribution(const LightRef & lr, float value)
{
  assert(lr.type >= 0 && lr.type < NUM_LIGHT_TYPES);
  light_sampler[lr.type].ObserveReturn(lr.idx, value);
}


void UcbLightPickerImpl::UcbLightPicker::UpdateAllShared()
{
  int i = 0;
  for (auto &s : light_sampler)
  {
    s.UpdateAllShared(/*weight_sum = */light_type_selection_probs[i++]);
  }
  light_type_selection_probs /= light_type_selection_probs.sum();
  for (auto &picker : CircularLinkList<UcbLightPicker>::rangeStartingAt(*this))
  {
    picker.light_type_selection_probs = this->light_type_selection_probs;
  }
}


void UcbLightPickerImpl::UcbLightPicker::Print(std::ostream & os) const
{
  os << "Light selection probabilities: \n";
  auto PrintType = [this,&os](int idx, const char* name) {
    os << strformat("-- %s: %s --\n", name, light_type_selection_probs[idx]);
    light_sampler[idx].Print(os);
  };
  PrintType(LightPickerCommon::IDX_PROB_POINT, "IDX_PROB_POINT");
  PrintType(LightPickerCommon::IDX_PROB_AREA, "IDX_PROB_AREA");
  PrintType(LightPickerCommon::IDX_PROB_ENV, "IDX_PROB_ENV");
  PrintType(LightPickerCommon::IDX_PROB_VOLUME, "IDX_PROB_VOLUME");
}


}
#include "lightpicker_ucb.hxx"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/flow_graph.h>
#include <numeric>

#include "gtest/gtest.h"


namespace Lightpickers
{


namespace {
void ComputeDistributionAndUpdate(Stats* stats_by_light_type, LightSelectionProbabilityMap &distribution)
{
  for (int light_type = 0; light_type < Lights::NUM_LIGHT_TYPES; ++light_type)
  {
    auto &s = stats_by_light_type[light_type];
    s.ComputeDistributionAndUpdate(distribution.cummulative_probs[light_type], /*weight_sum = */distribution.light_type_selection_probs[light_type]);
  }
  distribution.light_type_selection_probs /= distribution.light_type_selection_probs.sum();
}
}




Stats::Stats() :
    accum(0)
{
}


void Stats::Init(int arm_count)
{
  step_count = 0;
  accum = decltype(accum)(arm_count);
}


void Stats::ComputeDistributionAndUpdate(Eigen::ArrayXd &cummulative_probs, double &out_weight_sum)
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

  const auto &count = accum.Counts();
  const auto &mean = accum.Mean();
  const auto &var = accum.Var(/*fill_value=*/large_weight);

  for (int i = 0; i < arm_count; ++i)
  {
      if (count[i] >= 1)
      {
        auto confidence_bound = std::sqrt(2.* logt / count[i]);
        cummulative_probs[i] = mean[i] + std::sqrt(var[i]/count[i]) + confidence_bound;
      }
      else
        cummulative_probs[i] = large_weight;
      assert(std::isfinite(cummulative_probs[i]));
  }
  TowerSamplingInplaceCumSum(AsSpan(cummulative_probs));
  out_weight_sum = cummulative_probs[arm_count - 1];
  assert(std::isfinite(out_weight_sum) && out_weight_sum > 0.);
  TowerSamplingNormalize(AsSpan(cummulative_probs));
}





UcbLightPicker::UcbLightPicker(const Scene & scene)
  : LightPickerCommon{ scene },
  distribution{ scene }
{
  const auto counts = GetNumLightTypes(scene);
  int i = 0;
  for (int lightcount : counts)
  {
    stats[i++].Init(lightcount);
  }
}


void UcbLightPicker::ObserveReturns(Span<const std::pair<LightRef, float>> buffer)
{
  // This should maybe be the integral over the full spectrum, to get the total power.
  // Instead I sort of take the average over the spectrum. It is not the same because
  // this way I'm factoring in the variance across the spectrum. Not sure if this is right ...
  for (const auto [lr, value] : buffer)
  {
    stats[lr.type].ObserveReturn(lr.idx, value);
    assert(std::isfinite(value));
  }
}


void Lightpickers::UcbLightPicker::ComputeDistribution()
{
  ComputeDistributionAndUpdate(stats, distribution);
}



/////////////////////////////////////////////////////////////////////////////////////////
/// Photon based light picker
/////////////////////////////////////////////////////////////////////////////////////////

PhotonUcbLightPicker::PhotonUcbLightPicker(const Scene &scene)
  : LightPickerCommon{ scene },
  distribution{ scene }
{
  const auto counts = GetNumLightTypes(scene);
  int i = 0;
  for (int lightcount : counts)
  {
    stats[i++].Init(lightcount);
  }
}


void PhotonUcbLightPicker::OnPassStart(const Span<LightRef> emitters_of_paths)
{
  // Reallocate because there is no simpler way to clear or resize the array.
  ucb_photon_path_returns = decltype(ucb_photon_path_returns)(emitters_of_paths.size());
}

void PhotonUcbLightPicker::OnPassEnd(const Span<LightRef> emitters_of_paths)
{
  for (int i = 0; i < emitters_of_paths.size(); ++i)
  {
    stats[emitters_of_paths[i].type].ObserveReturn(
      emitters_of_paths[i].idx, 
      ucb_photon_path_returns[i]);
  }
}

void PhotonUcbLightPicker::ObserveReturns(Span<const std::pair<int, float>> buffer)
{
  for (auto [path_index, value] : buffer)
  {
    assert(std::isfinite(ucb_photon_path_returns[path_index]));
    ucb_photon_path_returns[path_index] += value;
    assert(std::isfinite(ucb_photon_path_returns[path_index]));
  }
}




void PhotonUcbLightPicker::ComputeDistribution()
{
  ComputeDistributionAndUpdate(stats, distribution);
}


}
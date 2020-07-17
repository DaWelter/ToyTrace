#include "lightpicker_ucb.hxx"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/flow_graph.h>
#include <numeric>

#include "gtest/gtest.h"


namespace Lightpickers
{


std::array<int, Lights::NUM_LIGHT_TYPES> GetNumLightTypes(const Scene & scene)
{
  std::array<int, Lights::NUM_LIGHT_TYPES> ret;
  ret[IDX_PROB_POINT] = scene.GetNumPointLights();
  ret[IDX_PROB_AREA] = scene.GetNumAreaLights();
  ret[IDX_PROB_ENV] = scene.HasEnvLight() ? 1 : 0;
  ret[IDX_PROB_VOLUME] = 0;
  return ret;
}


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




LightSelectionProbabilityMap::LightSelectionProbabilityMap(const Scene & scene)
  : scene{ scene }
{
  const auto counts = GetNumLightTypes(scene);

  for (int i = 0; i < NUM_LIGHT_TYPES; ++i)
  {
    cummulative_probs[i] = Eigen::VectorXd(counts[i]);
    if (counts[i] > 0)
    {
      cummulative_probs[i].setConstant(1. / counts[i]); // Uniform distribution
      TowerSamplingComputeNormalizedCumSum(AsSpan(cummulative_probs[i]));
    }
  }
  light_type_selection_probs = Eigen::Map<const Eigen::ArrayXi>(counts.data(), Eigen::Index(counts.size())).cast<double>();
  light_type_selection_probs /= light_type_selection_probs.sum();
}

void Lightpickers::LightSelectionProbabilityMap::Print(std::ostream & os) const
{
  os << "Light selection probabilities: \n";
  auto PrintType = [this, &os](int t, const char* name) {
    os << fmt::format("-- {}: {} --\n", name, light_type_selection_probs[t]);
    for (int i = 0; i<cummulative_probs[t].size(); ++i)
      os << fmt::format("p[{}]={}\n", i, TowerSamplingProbabilityFromCmf(AsSpan(cummulative_probs[t]), i));
  };
  PrintType(IDX_PROB_POINT, "IDX_PROB_POINT");
  PrintType(IDX_PROB_AREA, "IDX_PROB_AREA");
  PrintType(IDX_PROB_ENV, "IDX_PROB_ENV");
  PrintType(IDX_PROB_VOLUME, "IDX_PROB_VOLUME");
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
    ucb_photon_path_returns[path_index] += value;
  }
}




void PhotonUcbLightPicker::ComputeDistribution()
{
  ComputeDistributionAndUpdate(stats, distribution);
}


//void SetupGraph(tbb::flow::graph &g)
//{
//  tbb::flow::function_node<LightRefReturnsBuffer, tbb::flow::continue_msg> func(g, 1, [&](LightRefReturnsBuffer &buffer) {
//    // fill
//  });
//}




}
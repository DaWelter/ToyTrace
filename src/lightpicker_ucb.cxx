#include "lightpicker_ucb.hxx"

#if 0
void UCBStyleSampler::Init(int arm_count)
{
  mean = decltype(mean)::Zero(arm_count);
  sqr_sum = decltype(sqr_sum)::Zero(arm_count);
  count = decltype(count)::Zero(arm_count);
  cummulative_probs = decltype(cummulative_probs)::Zero(arm_count);
  step_count = 0;
}

void UCBStyleSampler::ObserveReturn(int arm, double value)
{
  OnlineVariance::Update(mean[arm], sqr_sum[arm], count[arm], value);
}

void UCBStyleSampler::UpdateProbs()
{
  ++step_count;
  const double large_weight = std::numeric_limits<double>::max() / mean.rows() / 10; // To make it very likely that unsampled arms are visited.
  const double logt = std::log(step_count);
  tbb::parallel_for(tbb::blocked_range<int>(0, (int)mean.rows(), 64), [this, logt, large_weight](const tbb::blocked_range<int> &r) {
    for (int i = r.begin(); i < r.end(); ++i)
    {
      if (count[i] >= 2)
      {
        auto var = OnlineVariance::FinalizeVariance(sqr_sum[i], count[i]);
        auto confidence_bound = std::sqrt((3. / 2.)*var*logt / count[i]);
        cummulative_probs[i] =
          mean[i] + confidence_bound;
        assert(std::isfinite(cummulative_probs[i]));
      }
      else
      {
        cummulative_probs[i] = large_weight;
      }
    }
  });
  TowerSamplingComputeNormalizedCumSum(Span<double>(cummulative_probs.data(), cummulative_probs.rows()));
}

Span<const double> UCBStyleSampler::GetCummulativeProbs()
{
  return Span<const double>(cummulative_probs.data(), cummulative_probs.rows());
}



UcbLightPicker::UcbLightPicker(const Scene & scene)
  : scene{ scene }
{
  // Copy paste from trivial light picker. First get all area light primitives.
  for (int geom_idx = 0; geom_idx < scene.GetNumGeometries(); ++geom_idx)
  {
    const auto &geom = scene.GetGeometry(geom_idx);
    for (int prim_idx = 0; prim_idx < geom.Size(); ++prim_idx)
    {
      auto &mat = scene.GetMaterialOf(geom_idx, prim_idx);
      if (mat.emitter)
      {
        arealight_refs.push_back(std::make_pair(geom_idx, prim_idx));
      }
    }
  }
  // Then get volumetric emitters
  for (int i = 0; i < scene.GetNumMaterials(); ++i)
  {
    const auto *medium = scene.GetMaterial(i).medium;
    if (medium && medium->is_emissive)
      volume_light_refs.push_back(i);
  }
  // Setup the UcbSamplers

}

void UcbLightPicker::PickLight(Sampler & sampler, PickCallbacks &cb) const
{
}

void UcbLightPicker::ObserveLightContribution(const LightRef & lr, const Spectral3 & radiance)
{
}
#endif
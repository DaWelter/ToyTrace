#include "lightpicker_trivial.hxx"

namespace Lightpickers
{

LightPickerCommon::LightPickerCommon(const Scene & scene)
  : scene{ scene }
{
}


TrivialLightPicker::TrivialLightPicker(const Scene& _scene)
  : LightPickerCommon{ _scene }
{
  const auto nl = (double)scene.GetNumPointLights();
  const auto ne = (double)(scene.HasEnvLight() ? 1 : 0);
  const auto na = (double)(scene.GetNumAreaLights());
  const auto nv = 0; // (double)(scene.GetNumVolumeLights());
  // Why do I need to initialize like this to not get a negative number in IDX_PROB_POINT?
  // I mean when nl is zero and I assign an initializer list, the last entry is going to be like -something.e-42. Why???
  const double normalize_factor = 1. / (nl + ne + na + nv);
  emitter_type_selection_probabilities[IDX_PROB_ENV] = ne * normalize_factor;
  emitter_type_selection_probabilities[IDX_PROB_AREA] = na * normalize_factor;
  emitter_type_selection_probabilities[IDX_PROB_POINT] = nl * normalize_factor;
  emitter_type_selection_probabilities[IDX_PROB_VOLUME] = nv * normalize_factor;
  in_class_probabilities[IDX_PROB_ENV] = ne > 0 ? (1. / ne) : NaN;
  in_class_probabilities[IDX_PROB_AREA] = na > 0 ? (1. / na) : NaN;
  in_class_probabilities[IDX_PROB_POINT] = nl > 0 ? (1. / nl) : NaN;
  in_class_probabilities[IDX_PROB_VOLUME] = 0; // Not implemented
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



std::array<int, Lights::NUM_LIGHT_TYPES> GetNumLightTypes(const Scene & scene)
{
  std::array<int, Lights::NUM_LIGHT_TYPES> ret;
  ret[IDX_PROB_POINT] = scene.GetNumPointLights();
  ret[IDX_PROB_AREA] = scene.GetNumAreaLights();
  ret[IDX_PROB_ENV] = scene.HasEnvLight() ? 1 : 0;
  ret[IDX_PROB_VOLUME] = 0;
  return ret;
}




}
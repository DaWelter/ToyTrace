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

}
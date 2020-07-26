#include "light.hxx"
#include "sampler.hxx"
#include "scene.hxx"
#include "rendering_util.hxx"

#include <tbb/parallel_for.h>

namespace RadianceOrImportance 
{
 
TotalEnvironmentalRadianceField::TotalEnvironmentalRadianceField(const IndividualLights &envlights_) 
  : envlights{ envlights_ } 
{
  if (envlights.size() > std::numeric_limits<int>::max())
    throw std::range_error("Cannot handle that many env lights!");
}


inline const EnvironmentalRadianceField& TotalEnvironmentalRadianceField::get(int i) const 
{ 
  return *envlights[i];
}

inline int TotalEnvironmentalRadianceField::size() const 
{ 
  return (int)envlights.size(); 
}
  
DirectionalSample TotalEnvironmentalRadianceField::TakeDirectionSample(Sampler& sampler, const PathContext& context) const
{
    assert(size()>0);
    int idx_sample = sampler.UniformInt(0, size()-1);
    const double selection_probability = 1./size();
    DirectionalSample smpl = get(idx_sample).TakeDirectionSample(sampler, context);
#if 1
    // This simple version seems to work just as well.
    smpl.pdf_or_pmf *= selection_probability; // Russian roulette style removal of all but one sub-component.
    return smpl;
#else
    if (IsFromPmf(smpl))
    {
        smpl.pdf_or_pmf *= selection_probability;
        return smpl;
    }
    else
    {
        double total_pdf = smpl.pdf_or_pmf;
        for (int i=0; i<size(); ++i)
        {
            if (i == idx_sample) continue;
            smpl.value += get(i).Evaluate(smpl.coordinates, context);
            total_pdf += get(i).EvaluatePdf(smpl.coordinates, context);
        }
        smpl.pdf_or_pmf = Pdf {selection_probability*total_pdf}; // TODO: This cannot be right.
    }
    return smpl;
#endif
}


Spectral3 TotalEnvironmentalRadianceField::Evaluate(const Double3& emission_dir, const PathContext& context) const
{
    Spectral3 environmental_radiance {0.};
    for (int i=0; i<size(); ++i)
    {
        const auto &light = get(i);
        environmental_radiance += light.Evaluate(
                                      emission_dir, context);
    }
    return environmental_radiance;
}


double TotalEnvironmentalRadianceField::EvaluatePdf(const Double3& dir_out, const PathContext& context) const
{
    const double selection_probability = size()>0 ? 1./size() : 1.;
    double pdf_sum = 0.;
    for (int i=0; i<size(); ++i)
    {
        const auto &light = get(i);
        pdf_sum += light.EvaluatePdf(dir_out, context);
    }
    return pdf_sum * selection_probability;
}



EnvMapLight::EnvMapLight(const Texture* texture_, const Double3 &up_dir_)
  : frame{OrthogonalSystemZAligned(up_dir_.cast<float>())}, texture{texture_}
{

  const int w = texture->Width();
  const int h = texture->Height();
  const int num_pixels = h*w;
  
  cmf_rows.resize(h);
  cmf_cols.resize(h, w);

  tbb::parallel_for(0, h, [w,this](int y)
  {
    double row_weight = 0.;
    for (int x=0; x<w; ++x)
    {
      RGB col = texture->GetPixel(x, y);
      Float2 angles = Projections::UvToSpherical(PixelCenterToUv(*texture, {x, y}));
      double weight = std::sin(angles[1])*(double)col.mean(); // Because differential solid angle is dS = sin(theta)*dtheta*dphi
      // We neglect the integration over pixels because we have piecewise constant uniform grid, so 
      // the delta_theta*delta_phi factors are the same for every pixel.
      row_weight += weight;
      cmf_cols(y, x) = row_weight; // Cumulative sum.
    }
    cmf_cols.row(y) /= row_weight;
    cmf_rows[y] = row_weight;
  });
  TowerSamplingComputeNormalizedCumSum(AsSpan(cmf_rows));
}


std::pair<int,int> EnvMapLight::MapToImage(const Double3 &dir_out) const
{
  auto uv = Projections::SphericalToUv(Projections::KartesianToSpherical(frame.transpose()*dir_out.cast<float>()));
  return UvToPixel(*texture, uv.cast<float>());
}


#define ENV_MAP_IMPORTANCE_SAMPLING 1


DirectionalSample EnvMapLight::TakeDirectionSample(Sampler &sampler, const PathContext &context) const
{
#if ENV_MAP_IMPORTANCE_SAMPLING
    const Double2 r = sampler.UniformUnitSquare();
    const auto y = TowerSamplingBisection<float>(AsSpan(cmf_rows), r[1]);
    const float proby = TowerSamplingProbabilityFromCmf(AsSpan(cmf_rows), y);
    assert(0 <= y && y < texture->Height());
    const auto rowspan = RowSpan(cmf_cols,y);
    const auto x = TowerSamplingBisection<float>(rowspan, r[0]);
    const float probx = TowerSamplingProbabilityFromCmf(rowspan, x);
    assert(0 <= x && x < texture->Width());
    auto uv_bounds = PixelToUvBounds(*texture,{x, y});
    Float2 angles_lower = Projections::UvToSpherical(uv_bounds.first);
    Float2 angles_upper = Projections::UvToSpherical(uv_bounds.second);
    const double z0 = std::cos(angles_lower[1]);
    const double z1 = std::cos(angles_upper[1]);
    Float3 dir_out = frame*SampleTrafo::ToUniformSphereSection(sampler.GetRandGen().UniformUnitSquare(), angles_lower[0], z0, angles_upper[0], z1).cast<float>();
    ASSERT_NORMALIZED(dir_out);
    double pdf = proby*probx;
    pdf /= (z1-z0)*(angles_upper[0]-angles_lower[0]);
    assert(pdf > 0);
    Spectral3 col = Color::RGBToSpectralSelection(texture->GetPixel(x,y), context.lambda_idx);
#else  // No importance sampling
    auto dir_out = frame.cast<double>() * SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
    auto pdf = EvaluatePdf(dir_out, context);
    auto col = Evaluate(dir_out, context);
#endif
    DirectionalSample s {
      dir_out.cast<double>(),
      col,
      pdf };
    return s;
}


Spectral3 EnvMapLight::Evaluate(const Double3 &dir_out, const PathContext &context) const
{
  auto pix = MapToImage(dir_out);
  RGB col = texture->GetPixel(pix);
  return Color::RGBToSpectralSelection(col, context.lambda_idx);
}


double EnvMapLight::EvaluatePdf(const Double3 &dir_out, const PathContext &context) const
{
#if ENV_MAP_IMPORTANCE_SAMPLING
  auto [x,y] = MapToImage(dir_out);
  const float proby = TowerSamplingProbabilityFromCmf(AsSpan(cmf_rows), y);
  const float probx = TowerSamplingProbabilityFromCmf(RowSpan(cmf_cols,y), x);
  auto uv_bounds = PixelToUvBounds(*texture, {x,y});
  Float2 angles_lower = Projections::UvToSpherical(uv_bounds.first);
  Float2 angles_upper = Projections::UvToSpherical(uv_bounds.second);
  const double z0 = std::cos(angles_lower[1]);
  const double z1 = std::cos(angles_upper[1]);
  double pdf = proby*probx;
  pdf /= (z1-z0)*(angles_upper[0]-angles_lower[0]);
  return pdf;
#else
  return 1./UnitSphereSurfaceArea;
#endif
}





} // namespace RadianceOrImportance


namespace Lights {
namespace Detail {

EnvLightPointSamplingBeyondScene::EnvLightPointSamplingBeyondScene(const Scene & scene)
{
  Box bb = scene.GetBoundingBox();
  box_center = 0.5*(bb.max + bb.min);
  diameter = Length(bb.max - bb.min);
  sufficiently_long_distance_to_go_outside_the_scene_bounds = 10.*diameter;
}


Double3 EnvLightPointSamplingBeyondScene::Sample(const Double3 & exitant_direction, Sampler & sampler) const
{
  Double3 disc_sample = SampleTrafo::ToUniformDisc(sampler.UniformUnitSquare());
  Eigen::Matrix3d frame = OrthogonalSystemZAligned(exitant_direction);
  Double3 org =
    box_center +
    -sufficiently_long_distance_to_go_outside_the_scene_bounds * exitant_direction
    + frame * 0.5 * diameter * disc_sample;
  return org;
}



RaySegment SegmentToEnv(const SurfaceInteraction & surface, const Scene & scene, const Double3 & dir)
{
  Ray ray{ surface.pos, -dir };
  const double length = 10.*scene.GetBoundingBox().DiagonalLength();
  ray.org += AntiSelfIntersectionOffset(surface, ray.dir);
  return { ray, length };
}

RaySegment SegmentToEnv(const VolumeInteraction & volume, const Scene & scene, const Double3 & dir)
{
  Ray ray{ volume.pos, -dir };
  const double length = 10.*scene.GetBoundingBox().DiagonalLength();
  return { ray, length };
}


RaySegment SegmentToPoint(const SurfaceInteraction & surface, const Double3 & pos)
{
  Ray ray{ surface.pos, pos - surface.pos };
  double length = Length(ray.dir);
  if (length > 0.)
    ray.dir /= length;
  else // Just pick something which will not result in NaN.
  {
    length = Epsilon;
    ray.dir = surface.normal;
  }
  ray.org += AntiSelfIntersectionOffset(surface, ray.dir);
  return { ray, length };
}

RaySegment SegmentToPoint(const VolumeInteraction& volume, const Double3 & pos)
{
  Ray ray{ volume.pos, pos - volume.pos };
  double length = Length(ray.dir);
  if (length > 0.)
    ray.dir /= length;
  else // Just pick something which will not result in NaN.
  {
    length = Epsilon;
    ray.dir = Double3{ 1.,0.,0. };
  }
  return { ray, length };
}




}  // Lights::Detail::


LightConnectionSample Area::SampleConnection(const SomeInteraction & somewhere, const Scene & scene, Sampler & sampler, const PathContext & context)
{
  auto area_smpl = emitter->TakeAreaSample(prim_ref, sampler, context);
  this->light_surf = SurfaceInteraction{ area_smpl.coordinates };
  const auto seg = mpark::visit([this](auto && ia) -> RaySegment {
    return Detail::SegmentToPoint(ia, light_surf.pos);
  }, somewhere);
  const auto val = emitter->Evaluate(area_smpl.coordinates, -seg.ray.dir, context, nullptr);
  const auto dfactor = DFactorPBRT(light_surf, seg.ray.dir);
  return { seg, area_smpl.pdf_or_pmf, dfactor*val };
}


}
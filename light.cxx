#include "light.hxx"
#include "scene.hxx"

namespace RadianceOrImportance 
{
  
inline const EnvironmentalRadianceField& TotalEnvironmentalRadianceField::get(int i) const 
{ 
  return scene.GetEnvLight(i);
}

inline int TotalEnvironmentalRadianceField::size() const 
{ 
  return scene.GetNumEnvLights(); 
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



EnvMapLight::EnvMapLight(const Texture* _texture, const Double3 &_up_dir)
  : texture{_texture}
{
  frame = OrthogonalSystemZAligned(_up_dir.cast<float>());
  int num_pixels = texture->Height()*texture->Width();
  cmf.resize(num_pixels);
  
  const int w = texture->Width();
  const int h = texture->Height();
  for (int y=0; y<h; ++y)
  {
    for (int x=0; x<w; ++x)
    {
      RGB col = texture->GetPixel(x, y);
      Float2 angles = Projections::UvToSpherical(PixelCenterToUv(*texture, {x, y}));
      double weight = std::sin(angles[1]); // Because differential solid angle is dS = sin(theta)*dtheta*dphi
      // We neglect the integration over pixels because we have piecewise constant uniform grid, so 
      // the delta_theta*delta_phi factors are the same for every pixel.
      cmf[RowMajorOffset(x, y, w, h)] = weight*(double)col.mean();
    }
  }
  TowerSamplingComputeNormalizedCumSum(AsSpan(cmf));
}


std::pair<int,int> EnvMapLight::MapToImage(const Double3 &dir_out) const
{
  auto uv = Projections::SphericalToUv(Projections::KartesianToSpherical(frame.transpose()*dir_out.cast<float>()));
  return UvToPixel(*texture, uv.cast<float>());
}


#define ENV_MAP_IMPORTANCE_SAMPLING 1


DirectionalSample EnvMapLight::TakeDirectionSample(Sampler &sampler, const PathContext &context) const
{
#if ENV_MAP_IMPORTANCE_SAMPLING // No importance sampling
    int pixel_index = TowerSamplingBisection(AsSpan(cmf), sampler.Uniform01());
    int x, y; std::tie(x,y) = RowMajorPixel(pixel_index, texture->Width(), texture->Height());
    auto uv_bounds = PixelToUvBounds(*texture,{x, y});
    Float2 angles_lower = Projections::UvToSpherical(uv_bounds.first);
    Float2 angles_upper = Projections::UvToSpherical(uv_bounds.second);
    const double z0 = std::cos(angles_lower[1]);
    const double z1 = std::cos(angles_upper[1]);
    Float3 dir_out = frame*SampleTrafo::ToUniformSphereSection(sampler.UniformUnitSquare(), angles_lower[0], z0, angles_upper[0], z1).cast<float>();
    ASSERT_NORMALIZED(dir_out);
    double pdf = TowerSamplingProbabilityFromCmf(AsSpan(cmf), pixel_index);
    pdf /= (z1-z0)*(angles_upper[0]-angles_lower[0]);
    assert(pdf > 0);
    Spectral3 col = Color::RGBToSpectralSelection(texture->GetPixel(x,y), context.lambda_idx);
#else
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
  auto pix = MapToImage(dir_out);
  int pixel_index = RowMajorOffset(pix.first, pix.second, texture->Width(), texture->Height());
  double pdf = TowerSamplingProbabilityFromCmf(AsSpan(cmf), pixel_index);
  auto uv_bounds = PixelToUvBounds(*texture, pix);
  Float2 angles_lower = Projections::UvToSpherical(uv_bounds.first);
  Float2 angles_upper = Projections::UvToSpherical(uv_bounds.second);
  const double z0 = std::cos(angles_lower[1]);
  const double z1 = std::cos(angles_upper[1]);
  pdf /= (z1-z0)*(angles_upper[0]-angles_lower[0]);
  return pdf;
#else
  return 1./UnitSphereSurfaceArea;
#endif
}





} // namespace 

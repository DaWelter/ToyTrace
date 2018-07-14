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
  frame = OrthogonalSystemZAligned(_up_dir);
  int num_pixels = texture->Height()*texture->Width();
  cmf.resize(num_pixels);
}

DirectionalSample EnvMapLight::TakeDirectionSample(Sampler &sampler, const PathContext &context) const
{
    auto dir_out = frame * SampleTrafo::ToUniformSphere(sampler.UniformUnitSquare());
    auto pdf = EvaluatePdf(dir_out, context);
    auto col = Evaluate(dir_out, context);
    DirectionalSample s {
      dir_out,
      col,
      pdf };
    return s;
}

Spectral3 EnvMapLight::Evaluate(const Double3 &dir_out, const PathContext &context) const
{
  auto uv = ToSphericalCoordinates(frame.transpose()*dir_out);
  RGB col = texture->GetTexel(uv.cast<float>());
  return Color::RGBToSpectralSelection(col, context.lambda_idx);
}

double EnvMapLight::EvaluatePdf(const Double3 &dir_out, const PathContext &context) const
{
  return 1./UnitSphereSurfaceArea;
}





} // namespace 

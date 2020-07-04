#pragma once

#include "lightpicker_trivial.hxx"
#include "light.hxx"

#include <tbb/task_arena.h>

namespace Lightpickers
{

#if 0
template<class T, class Derived, size_t buffer_size>
class BufferedOps
{
  using Item_ = T;
  std::unique_ptr<Item_[]> buffer;
  size_t count = 0;
  tbb::task_arena &arena;

  void EnqueueWorkInBuffer()
  {
    arena.enqueue([std::move(buffer)]{
      Derived::Work(std::move(buffer), count);
      });
  }

public:
  BufferedOps(tbb::task_arena &arena)
    : arena{ arena }
  {
    buffer.reset(new Item[buffer_size]);
  }

  ~BufferedOps()
  {
    assert(count == 0);
  }
  
  void Finish()
  {
    if (count > 0)
      EnqueueWorkInBuffer()
  }

  template<class U>
  void AddWork(const U && x)
  {
    buffer[count++] = Derived::ParallelProcessing(std::forward(x));
    if (count >= buffer_size)
    {
      EnqueueWorkInBuffer(arena);

      buffer.reset(new Item_[buffer_size]);
      count = 0;
    }
  }
};
#endif


class LightSelectionProbabilityMap
{
public:
    LightSelectionProbabilityMap(const Scene &scene);

    double Pmf(const LightRef &lr) const
    {
      return light_type_selection_probs[lr.type] * 
        TowerSamplingProbabilityFromCmf(AsSpan(cummulative_probs[lr.type]), lr.idx);
    }

    template<class Visitor>
    void Sample(Sampler &sampler, Visitor &&visitor) const;

    void Print(std::ostream &os) const;

    const Scene &scene;
    std::array<Eigen::ArrayXd, Lights::NUM_LIGHT_TYPES> cummulative_probs;
    Eigen::Array<double, 4, 1> light_type_selection_probs;
};

std::array<int, Lights::NUM_LIGHT_TYPES> GetNumLightTypes(const Scene &scene);


// Accumulate statistics about radiance received from each light.
class Stats
{
public:
  Stats();
  void Init(int arm_count);
  void ObserveReturn(int arm, double value)
  {
    accum.Add(arm, value);
  }
  void ComputeDistributionAndUpdate(Eigen::ArrayXd &cummulative, double &out_weight_sum);
  int ArmCount() const {
    return accum.Size();
  }
private:
  Accumulators::SoaOnlineVariance<double> accum;
  int step_count = 0;
};


class UcbLightPicker : public LightPickerCommon
{
public:
  UcbLightPicker(const Scene &scene);
  
  void ObserveReturns(Span<const std::pair<LightRef, float>> buffer);
  void ComputeDistribution();

  const auto &Distribution() const { return distribution; }

private:
  Stats stats[Lights::NUM_LIGHT_TYPES];
  LightSelectionProbabilityMap distribution;
};


class PhotonUcbLightPicker : public LightPickerCommon
{
public:
    PhotonUcbLightPicker(const Scene &scene);

    void OnPassStart(const Span<LightRef> emitters_of_paths);
    void ObserveReturns(Span<const std::pair<int, float>> buffer);
    void OnPassEnd(const Span<LightRef> emitters_of_paths);
    void ComputeDistribution();

    const auto &Distribution() const { return distribution; }

private:
    ToyVector<float> ucb_photon_path_returns;
    Stats stats[Lights::NUM_LIGHT_TYPES];
    LightSelectionProbabilityMap distribution;
};



//======================================================================
//======================================================================
//======================================================================
template<class Visitor>
inline void LightSelectionProbabilityMap::Sample(Sampler & sampler, Visitor && visitor) const
{
  const int which_kind = TowerSampling<NUM_LIGHT_TYPES>(
    light_type_selection_probs.data(), sampler.Uniform01());
  const int idx = TowerSamplingBisection<double>(AsSpan(cummulative_probs[which_kind]), sampler.Uniform01());
  const LightRef ref{ (uint32_t)which_kind, (uint32_t)idx };

  const double prob = Pmf(ref);

  switch (which_kind)
  {
  case IDX_PROB_ENV:
  {
    visitor(Lights::Env{ scene.GetTotalEnvLight() }, prob, ref);
  }
  break;
  case IDX_PROB_POINT:
  {

    visitor(Lights::Point{ scene.GetPointLight(idx) }, prob, ref);
  }
  break;
  case IDX_PROB_AREA:
  {
    auto prim_ref = scene.GetPrimitiveFromAreaLightIndex(idx);
    visitor(Lights::Area{ prim_ref, scene }, prob, ref);
  }
  break;
  case IDX_PROB_VOLUME:
  {
    assert(!"not implemented");
    //const int n = isize(volume_light_refs);
    //auto idx = volume_light_refs[sampler.UniformInt(0, n-1)];
    //double prob = emitter_type_selection_probabilities[which_kind]/n;
    //return visitor(*ASSERT_NOT_NULL(scene.GetMaterial(idx).medium), prob);
  }
  break;
  }
}

//======================================================================
//======================================================================
//======================================================================


inline float RadianceToObservationValue(const Spectral3 & r)
{
  return static_cast<float>(r.sum());
}

#if 0
class UcbLightPickerWorker : public BufferedOps<std::pair<LightRef, float>, UcbLightPickerWorker, 1024>
{
  UcbLightPicker &picker;
public:
  using Item = std::pair<LightRef, float>;

protected:
  void Work(std::unique_ptr<Item[]> items, int count)
  {
    picker.ObserveReturns(Span<Item>(items.get(), count));
  }

  std::pair<LightRef, float> ParallelProcessing(const std::pair<LightRef, Spectral3> &item)
  {
    return { item.first, RadianceToObservationValue(item.second) };
  }
};


class PhotonUcbLightPickerWorker : public BufferedOps<std::pair<int, float>, PhotonUcbLightPickerWorker, 1024>
{
  PhotonUcbLightPicker &picker;
public:
  using Item = std::pair<int, float>;

protected:
  void Work(std::unique_ptr<Item[]> items, int count)
  {
    picker.ObserveReturns(Span<Item>(items.get(), count));
  }

  std::pair<int, float> ParallelProcessing(const std::pair<int, Spectral3> &item)
  {
    return { item.first, RadianceToObservationValue(item.second) };
  }
};
#endif

} // namespace Lightpickers


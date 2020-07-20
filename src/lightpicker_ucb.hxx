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


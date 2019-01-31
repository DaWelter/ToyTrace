#pragma once

#include <functional>

#include <tbb/atomic.h>
#include <tbb/mutex.h>
#include <tbb/spin_mutex.h>
#include <tbb/parallel_for.h>

#include "scene.hxx"
#include "util.hxx"
#include "shader_util.hxx"
#include "camera.hxx"
#include "renderbuffer.hxx"
#include "util_thread.hxx"
#include "renderingalgorithms_interface.hxx"



struct SensorResponse
{
  int unit_index { -1 };
  RGB weight {};
  operator bool() const { return unit_index>=0; }
};


namespace SimplePixelByPixelRenderingDetails
{

// Want to increase the number of samples per pixel over time in order to  
// a) give a quick preview
// b) have less iteration and display overhead later on where the image changes much slower.
class SamplesPerPixelSchedule
{
  int spp = 1; // Samples per pixel
  int total_spp = 0; // Samples till now.
  int max_spp;

public:
  SamplesPerPixelSchedule(const RenderingParameters &render_params) 
    : max_spp{render_params.max_samples_per_pixel}
  {}
  
  void UpdateForNextPass()
  {
    total_spp += spp;
    if (spp < 256)
    {
      spp *= 2;
    }
    if (max_spp > 0 && total_spp + spp > max_spp)
    {
      spp = max_spp - total_spp;
    }
  }
  
  inline int GetPerIteration() const { return spp; }
  inline int GetTotal() const { return total_spp; }
};


class SamplesPerPixelScheduleConstant
{
  int spp = 1; // Samples per pixel
  int total_spp = 0; // Samples till now.
  int max_spp;

public:
  SamplesPerPixelScheduleConstant(const RenderingParameters &render_params) 
    : max_spp{render_params.max_samples_per_pixel}
  {}
  
  void UpdateForNextPass()
  {
    total_spp += spp;
    if (max_spp > 0 && total_spp >= max_spp)
    {
      spp = 0;
    }
  }
  
  inline int GetPerIteration() const { return spp; }
  inline int GetTotal() const { return total_spp; }
};



class Worker
{
  public: 
    virtual ~Worker() {}
    virtual RGB RenderPixel(int _pixel_index) = 0;
    virtual ToyVector<SensorResponse>& GetSensorResponses() = 0;
};


class SimplePixelByPixelRenderingAlgo : public RenderingAlgo
{
protected:
  const RenderingParameters &render_params;
  const Scene &scene;
private:
  static constexpr int PIXEL_STRIDE = 64 / sizeof(Spectral3); // Because false sharing.
  Spectral3ImageBuffer buffer;
  int num_threads = 1;
  int num_pixels = 0;
  SamplesPerPixelSchedule spp_schedule;
  tbb::atomic<int> shared_pixel_index = 0;
  tbb::task_group the_task_group;
  tbb::atomic<bool> stop_flag = false;
  tbb::spin_mutex buffer_mutex;
  ToyVector<std::unique_ptr<Worker>> workers;
public:
  SimplePixelByPixelRenderingAlgo(const RenderingParameters &render_params_, const Scene &scene_)
    : RenderingAlgo{}, render_params{render_params_}, scene{scene_},
      buffer{render_params_.width, render_params_.height}, num_threads{0},
      spp_schedule{render_params_}
  {
    num_pixels = render_params.width * render_params.height;
  }
  
  void Run() override
  {
    for (int i=0; i<std::max(1, this->render_params.num_threads); ++i)
      workers.push_back(AllocateWorker(i));
    num_threads = workers.size();
    int pass = 0;
    while (!stop_flag.load() && spp_schedule.GetPerIteration() > 0)
    {
      shared_pixel_index = 0;
      buffer.AddSampleCount(GetSamplesPerPixel());
      while_parallel_fed_interruptible(
        /*func=*/[this](int pixel_index, int worker_num)
        {
          this->RunRenderingWorker(pixel_index, *workers[worker_num]);
        },
        /*feeder=*/[this]() -> boost::optional<int>
        {
          return this->FeedPixelIndex();
        },
        /*irq_handler=*/[this]() -> bool
        {
          if (this->stop_flag.load())
            return false;
          this->CallInterruptCb(false);
          return true;
        },
        num_threads, the_task_group);
      std::cout << "Iteration finished, past spp = " << GetSamplesPerPixel() << ", total taken " << spp_schedule.GetTotal() << std::endl;
      PassCompleted();
      CallInterruptCb(true);
      spp_schedule.UpdateForNextPass();
    } // Pass iteration
  }
  
  // Will potentially be called before Run is invoked!
  void RequestFullStop() override
  {
    stop_flag.store(true);
    the_task_group.cancel();
  }
  
  // Will potentially be called before Run is invoked!
  void RequestInterrupt() override
  {
    the_task_group.cancel();
  }
  
  std::unique_ptr<Image> GenerateImage() override
  {
    auto bm = std::make_unique<Image>(render_params.width, render_params.height);
    tbb::parallel_for(0, render_params.height, [&](int row){
      buffer.ToImage(*bm, row, row+1);  
    });
    return bm;
  }
  
protected:
  // Implementation must override this.
  virtual std::unique_ptr<Worker> AllocateWorker(int num) = 0;

  virtual void PassCompleted() {};
  
  inline int GetNumPixels() const { return num_pixels; }
  
  inline int GetSamplesPerPixel() const { return spp_schedule.GetPerIteration(); }
  
private:
  boost::optional<int> FeedPixelIndex()
  {
    int i = shared_pixel_index.fetch_and_add(PIXEL_STRIDE);
    return i <= num_pixels ? i : boost::optional<int>();
  }
  
  
  void RunRenderingWorker(int pixel_index, Worker &worker)
  {
    RenderPixels(pixel_index, worker);
    if (worker.GetSensorResponses().size() > 0)
    { // Fill in the samples from light tracing.
      tbb::spin_mutex::scoped_lock lock(buffer_mutex);
      SplatLightSamples(worker);
    }
  }
  
  
  void RenderPixels(int pixel_index, Worker &worker)
  {
    int current_end_index = std::min(pixel_index + PIXEL_STRIDE, num_pixels);
    for (; pixel_index < current_end_index; ++pixel_index)
    {
      const int nsmpl = GetSamplesPerPixel();
      for(int i=0;i<nsmpl;i++)
      {
        auto smpl = worker.RenderPixel(pixel_index);
        buffer.Insert(pixel_index, smpl);
      }
    }    
  }
  
  
  void SplatLightSamples(Worker &worker)
  {
    for (const auto &r : worker.GetSensorResponses())
    {
      assert(r);
      buffer.Splat(r.unit_index, r.weight);
    }
    worker.GetSensorResponses().clear();
  }
};


} // namespace SimplePixelByPixelRenderingDetails




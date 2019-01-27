#pragma once

#include <functional>

#include <tbb/atomic.h>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>

#include "scene.hxx"
#include "util.hxx"
#include "shader_util.hxx"
#include "camera.hxx"
#include "renderbuffer.hxx"


using InterruptCallback = std::function<void(bool)>;


class RenderingAlgo
{
  InterruptCallback irq_cb;
public:
  virtual ~RenderingAlgo() {}
  // The callback will be invoked from the same thread where Run was invoked.
  void SetInterruptCallback(InterruptCallback cb) { irq_cb = cb; }
  virtual void Run() = 0;
  // Will cause the interrupt callback to be invoked after an unspecified amount of time.
  // For intermediate status reports, images, etc. 
  virtual void RequestInterrupt() = 0;
  virtual void RequestFullStop() = 0; // Early termination due to user interaction.
  // May be called from within the interrupt callback, or after Run returned.
  virtual std::unique_ptr<Image> GenerateImage() = 0;
protected:
  void CallInterruptCb(bool is_complete_pass) { irq_cb(is_complete_pass); }
};


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
    if (spp < 256)
    {
      spp *= 2;
    }
    if (max_spp > 0 && total_spp + spp > max_spp)
    {
      spp = max_spp - total_spp;
    }
    total_spp += spp;
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
  Spectral3ImageBuffer buffer;
  int num_threads = 1;
  int num_pixels = 0;
  SamplesPerPixelSchedule spp_schedule;
  tbb::atomic<int> shared_pixel_index = 0;
  enum ControlFlag : int 
  {
    RUN = 0,
    REQUEST_INTERRUPT = 1,
    REQUEST_STOP = 2
  };
  tbb::atomic<int> control_flag = RUN;
  tbb::mutex buffer_mutex;
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
    while (!(control_flag.load() & REQUEST_STOP) && spp_schedule.GetPerIteration() > 0)
    {
      shared_pixel_index = 0;
      buffer.AddSampleCount(GetSamplesPerPixel());
      while (shared_pixel_index.load() < num_pixels && !(control_flag.load() & REQUEST_STOP))
      {
        tbb::parallel_for(0, num_threads, 1, [this](int task_num) {
          this->RunRenderingWorker(*workers[task_num]);
        });
        if (control_flag.load() & REQUEST_INTERRUPT)
        {
          CallInterruptCb(false);
          control_flag.store(control_flag.load() & ~REQUEST_INTERRUPT);
        }
      } // Iteration over pixels
      //NotifyRenderingPassFinished();
      std::cout << "Iteration finished, spp = " << GetSamplesPerPixel() << ", total " << spp_schedule.GetTotal() << std::endl;
      CallInterruptCb(true);
      control_flag.store(control_flag.load() & ~REQUEST_INTERRUPT);
      spp_schedule.UpdateForNextPass();
    } // Pass iteration
  }
  
  // Will potentially be called before Run is invoked!
  void RequestFullStop() override
  {
    control_flag.store(REQUEST_STOP | control_flag.load());
  }
  
  // Will potentially be called before Run is invoked!
  void RequestInterrupt() override
  {
    control_flag.store(REQUEST_INTERRUPT | control_flag.load());
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

  inline int GetNumPixels() const { return num_pixels; }
  
  inline int GetSamplesPerPixel() const { return spp_schedule.GetPerIteration(); }
  
private:     
  void RunRenderingWorker(Worker &worker)
  {
    static constexpr int PIXEL_STRIDE = 64 / sizeof(Spectral3); // Because false sharing.
    int pixel_index = 0;
    while (control_flag.load() == RUN && (pixel_index = shared_pixel_index.fetch_and_add(PIXEL_STRIDE)) < num_pixels)
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
      
      { // Fill in the samples from light tracing.
        tbb::mutex::scoped_lock lock(buffer_mutex);
        for (const auto &r : worker.GetSensorResponses())
        {
          assert(r);
          buffer.Splat(r.unit_index, r.weight);
        }
      }
    }
  }
};


} // namespace SimplePixelByPixelRenderingDetails


std::unique_ptr<RenderingAlgo> RenderAlgorithmFactory(const Scene &_scene, const RenderingParameters &_params);





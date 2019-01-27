#pragma once

#include <functional>
#include <memory>

#include "image.hxx"
#include "scene.hxx"


using InterruptCallback = std::function<void(bool)>;


class RenderingAlgo
{
  InterruptCallback irq_cb;
public:
  virtual ~RenderingAlgo() noexcept(false) {}
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



std::unique_ptr<RenderingAlgo> RenderAlgorithmFactory(const Scene &_scene, const RenderingParameters &_params);

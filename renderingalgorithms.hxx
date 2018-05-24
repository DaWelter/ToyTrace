#pragma once

#include "scene.hxx"
#include "util.hxx"
#include "shader_util.hxx"
#include "camera.hxx"

class IRenderingAlgo
{
public:
  struct SensorResponse
  {
    int unit_index { -1 };
    RGB weight {};
    operator bool() const { return unit_index>=0; }
  };
  virtual ~IRenderingAlgo() {}
  virtual RGB MakePrettyPixel(int pixel_index) = 0;
  virtual ToyVector<SensorResponse> &GetSensorResponses() = 0;
  virtual void NotifyPassesFinished(int) {}
};


std::unique_ptr<IRenderingAlgo> RenderAlgorithmFactory(const Scene &_scene, const RenderingParameters &_params);
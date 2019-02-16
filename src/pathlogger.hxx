#pragma once

#ifdef HAVE_JSON

#include "vec3f.hxx"
#include "util.hxx"
#include "spectral.hxx"

#include <tbb/concurrent_queue.h>
#include <tbb/spin_mutex.h>
#include <tbb/tbb_thread.h>

#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

class Pathlogger
{
public:
  struct Node
  {
    Double3 position;
    Double3 incident_dir{0.};
    Double3 exitant_dir{0.};
    Double3 normal{0.};
    Spectral3 weight_before{0.}; // Scattering.
    Spectral3 weight_after{0.};
  };
private:
  using Path = ToyVector<Node>;
  Path path;
  static std::string filename;
  static tbb::concurrent_bounded_queue<Path> to_process;
  static std::unique_ptr<tbb::tbb_thread> io_thread;
  
  static void DoIo();
  
  static rapidjson::Value Array3ToJSON(const Eigen::Array<double, 3, 1> &v, rapidjson::Document::AllocatorType &alloc);
  
public:
  static void Init(const std::string &filename_);
  
  Pathlogger() {}
  
  void WritePath() 
  {
    to_process.push(path);
    path.clear();
  }
  
  void NewPath()
  {
    path.clear();
  }
  
  Node& AddNode()
  {
    path.emplace_back();
    return path.back();
  }
};

#endif

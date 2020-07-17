#pragma once

#ifdef HAVE_JSON

#include "vec3f.hxx"
#include "util.hxx"
#include "spectral.hxx"

#include <tbb/concurrent_queue.h>
#include <tbb/spin_mutex.h>
#include <tbb/tbb_thread.h>

#include <fstream>


namespace pathlogger_detail
{

struct Node
{
  Double3 position = Double3::Zero();
  Double3 exitant_dir = Double3::Zero();
  Double3 geom_normal = Double3::Zero();
  Spectral3 weight = Spectral3::Zero();
  Spectral3 transmission_weight_to_next = Spectral3::Zero();
  bool is_surface = false;
  bool is_specular = false;
  bool is_merge = false;
};

struct Contribution
{
  Spectral3 pixel_contribution;
  Index3 wavelengths;
};

struct Path
{
  ToyVector<Node> nodes;
  Contribution contribution;
};

class ProcessStage;

}


class Pathlogger;

class IncompletePaths
{
  using Path = pathlogger_detail::Path;
  static ToyVector<std::unique_ptr<pathlogger_detail::Path>> paths;
  static tbb::spin_mutex m;
  Path* current_path = nullptr;

public: 
  using Node = pathlogger_detail::Node;
  using Contribution = pathlogger_detail::Contribution;
  using SubPathHandle = std::pair<Path*, int>;

  static void Init();

  static void Clear()
  {
    tbb::spin_mutex::scoped_lock l{ m };
    decltype(paths){}.swap(paths);
    paths.reserve(1024 * 1024);
  }

  static void CopyToLogger(Pathlogger &logger, const SubPathHandle &h);

  void NewPath()
  {
    tbb::spin_mutex::scoped_lock l{ m };
    paths.push_back(std::make_unique<Path>());
    current_path = paths.back().get();
    current_path->nodes.reserve(16);
  }

  void PushNode()
  {
    current_path->nodes.emplace_back();
    assert(current_path->nodes.size() < 1000);
  }

  SubPathHandle GetHandle()
  {
    return { current_path, isize(current_path->nodes) };
  }

  void PopNode()
  {
    assert(!current_path->nodes.empty());
    current_path->nodes.pop_back();
  }

  Node& GetNode(int i)
  {
    return current_path->nodes[util::Modulus(i, isize(current_path->nodes))];
  }

  Contribution& GetContribution()
  {
    return current_path->contribution;
  }
};


class Pathlogger
{
public:
  using Node = pathlogger_detail::Node;
  using Path = pathlogger_detail::Path;
  using Contribution = pathlogger_detail::Contribution;
  friend class IncompletePaths;
private:
  Path path;
  static tbb::concurrent_bounded_queue<Path> to_process;
  static std::unique_ptr<tbb::tbb_thread> io_thread;
  static ToyVector<std::unique_ptr<pathlogger_detail::ProcessStage>> pipeline;
  static void ThreadFunction();

public:
  static void Init(const std::string &filename_);
  
  Pathlogger() {}
  
  void WritePath() 
  {
    to_process.push(path);
  }
  
  void NewPath()
  {
    path = {};
  }

  void PushNode()
  {
    path.nodes.emplace_back();
    assert(path.nodes.size() < 1000);
  }

  void PopNode()
  {
    assert(!path.nodes.empty());
    path.nodes.pop_back();
  }

  Node& GetNode(int i)
  {
    return path.nodes[util::Modulus(i, isize(path.nodes))];
  }

  Contribution& GetContribution()
  {
    return path.contribution;
  }
};

#endif

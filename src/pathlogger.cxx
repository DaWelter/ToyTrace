#include "pathlogger.hxx"
#include "json.hxx"
#include <boost/container/small_vector.hpp>

#ifdef HAVE_JSON

namespace pathlogger_detail
{
using namespace rapidjson_util;

struct PathComparatorGreater
{
  bool operator()(const Path& a, const Path &b) const
  {
    return a.contribution.pixel_contribution.maxCoeff() > b.contribution.pixel_contribution.maxCoeff();
  }
};

using PathsMessage = boost::container::small_vector<Path, 1, AlignedAllocator<Path, 16>>;


class ProcessStage
{
public:
  virtual void Process(PathsMessage &&p) = 0;
  virtual ~ProcessStage() {}
};


class Output : public ProcessStage
{
  std::string filename;
public:
  Output(const std::string &filename)
    : filename{ filename }
  {}

  void Process(PathsMessage &&paths) override
  {
    namespace rj = rapidjson;
    std::ofstream ofs{ filename, std::ios::trunc };
    for (const auto &p : paths)
    {
      rj::Document rjpath;
      auto &alloc = rjpath.GetAllocator();
      rjpath.SetObject();
      auto contribution_node = rj::Value{ rj::kObjectType };
      auto nodes_node = rj::Value{ rj::kArrayType };
      for (const auto &n : p.nodes)
      {
        rj::Value rjnode(rj::kObjectType);
        rjnode.AddMember("position", ToJSON(n.position.array(), alloc), alloc);
        rjnode.AddMember("geom_normal", ToJSON(n.geom_normal.array(), alloc), alloc);
        rjnode.AddMember("exitant_dir", ToJSON(n.exitant_dir.array(), alloc), alloc);
        rjnode.AddMember("weight", ToJSON(n.weight, alloc), alloc);
        rjnode.AddMember("transmission_weight_to_next", ToJSON(n.transmission_weight_to_next, alloc), alloc);
        rjnode.AddMember("is_specular", n.is_specular, alloc);
        rjnode.AddMember("is_surface", n.is_surface, alloc);
        rjnode.AddMember("is_merge", n.is_merge, alloc);
        nodes_node.PushBack(rjnode, alloc);
      }
      contribution_node.AddMember("pixel_value", ToJSON(p.contribution.pixel_contribution, alloc), alloc);
      contribution_node.AddMember("wavelengths", ToJSON(p.contribution.wavelengths, alloc), alloc);
      rjpath.AddMember("nodes", nodes_node, alloc);
      rjpath.AddMember("contribution", contribution_node, alloc);
      rj::StringBuffer buffer;
      rj::Writer<rj::StringBuffer> writer(buffer);
      rjpath.Accept(writer);
      ofs << buffer.GetString();
      ofs << "\n-----------------------------------------\n";
    }
  }
};


class FilterNLargest : public ProcessStage
{
  ToyVector<Path> minheap;
  long output_after = 100000;
  long paths_to_keep = 1024;
  long path_count = 0;
  ProcessStage* next = nullptr;
public:
  FilterNLargest(ProcessStage* next_) 
    : ProcessStage{}, next {next_ }
  {}

  void Process(PathsMessage &&paths) override
  {
    for (auto &p : paths)
    {
      if (lsize(minheap) >= paths_to_keep)
      {
        std::pop_heap(minheap.begin(), minheap.end(), PathComparatorGreater{});
        minheap.pop_back();
      }
      minheap.push_back(std::move(p));
      std::push_heap(minheap.begin(), minheap.end(), PathComparatorGreater{});

      ++path_count;
      if (path_count >= output_after)
      {
        PassToNextStage();
        path_count = 0;
      }
    }
  }

  void PassToNextStage()
  {
    PathsMessage msg;
    msg.insert(msg.begin(), minheap.begin(), minheap.end());
    next->Process(std::move(msg));
  }
};


}


tbb::concurrent_bounded_queue<Pathlogger::Path> Pathlogger::to_process;
std::unique_ptr<tbb::tbb_thread> Pathlogger::io_thread;
ToyVector<std::unique_ptr<pathlogger_detail::ProcessStage>> Pathlogger::pipeline;

void Pathlogger::ThreadFunction()
{
  pathlogger_detail::PathsMessage paths;
  while (true)
  {
    paths = pathlogger_detail::PathsMessage(1);
    to_process.pop(paths.back()); // Blocks
    pipeline.back()->Process(std::move(paths));
  }
}

void Pathlogger::Init(const std::string& filename)
{
  pipeline.push_back(std::make_unique<pathlogger_detail::Output>(filename));
  pipeline.push_back(std::make_unique<pathlogger_detail::FilterNLargest>(pipeline.back().get()));
  io_thread = std::make_unique<tbb::tbb_thread>(ThreadFunction); // Start
}


ToyVector<std::unique_ptr<pathlogger_detail::Path>> IncompletePaths::paths;
tbb::spin_mutex IncompletePaths::m;

void IncompletePaths::Init()
{
  paths.reserve(1024 * 1024);
}

void IncompletePaths::CopyToLogger(Pathlogger &logger, const SubPathHandle &h)
{
  logger.NewPath();
  const auto& path = *h.first;
  logger.path.contribution = path.contribution;
  for (int i=0; i<h.second; ++i)
  {
    logger.path.nodes.push_back(path.nodes[i]);
  }
}

#endif
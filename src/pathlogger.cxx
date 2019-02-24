#include "pathlogger.hxx"


std::string Pathlogger::filename;
tbb::concurrent_bounded_queue<Pathlogger::Path> Pathlogger::to_process;
std::unique_ptr<tbb::tbb_thread> Pathlogger::io_thread;

void Pathlogger::DoIo()
{
  namespace rj = rapidjson;
  std::ofstream ofs(filename);

  Path p;
  while (true)
  {
    to_process.pop(p); // Blocks
    std::cout << "dumping path" << std::endl;
    
    rj::Document rjpath;
    auto &alloc = rjpath.GetAllocator();
    rjpath.SetArray();

    //rj::Value rjpath(rj::kArrayType);
    for (const auto &n : p)
    {
      rj::Value rjnode(rj::kObjectType);
      rjnode.AddMember("position", Array3ToJSON(n.position.array(), alloc), alloc);
      rjnode.AddMember("normal", Array3ToJSON(n.normal.array(), alloc), alloc);
      rjnode.AddMember("geom_normal", Array3ToJSON(n.geom_normal.array(), alloc), alloc);
      rjnode.AddMember("incident_dir", Array3ToJSON(n.incident_dir.array(), alloc), alloc);
      rjnode.AddMember("exitant_dir", Array3ToJSON(n.exitant_dir.array(), alloc), alloc);
      rjnode.AddMember("weight_before", Array3ToJSON(n.weight_before, alloc), alloc);
      rjnode.AddMember("weight_after", Array3ToJSON(n.weight_after, alloc), alloc);
      rjnode.AddMember("is_surface", n.is_surface, alloc);
      rjpath.PushBack(rjnode, alloc);
    }

    rj::StringBuffer buffer;
    rj::Writer<rj::StringBuffer> writer(buffer);
    rjpath.Accept(writer);
    
    ofs << "-----------------------------------------\n";

    ofs << buffer.GetString();
    ofs << std::flush;
  }
}


rapidjson::Value Pathlogger::Array3ToJSON(const Eigen::Array< double, 3, 1 >& v, rapidjson::Document::AllocatorType& alloc)
{
  namespace rj = rapidjson;
  rj::Value json_vec(rj::kArrayType);
  json_vec.PushBack(rj::Value(v[0]).Move(), alloc).
  PushBack(rj::Value(v[1]).Move(), alloc).
  PushBack(rj::Value(v[2]).Move(), alloc);
  return json_vec;
}


void Pathlogger::Init(const std::string& filename_)
{
  filename = filename_;
  io_thread = std::make_unique<tbb::tbb_thread>(DoIo); // Start
}

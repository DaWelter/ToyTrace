#include <memory>
#include <memory_resource>

namespace util
{

// Worst case, this seems to be only two times faster than malloc.
// That is when 
// 1) writing to the new allocation causes a write to
// main memory - eviction of a cache line.
// 2) there is only one thread.
// 3) a benchmark situation where malloc only has to deal with exactly the 
//    same allocations than the arena.
// ???
class MemoryArena
{
  public:
    MemoryArena(size_t initial_size)
      : buffer_resource{initial_size}
      {}

    template<class T, class... Args>
    auto MakeUnique(Args&&... args)
    {
        auto* p = GetAllocator<T>().allocate(1);
        ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
        return std::unique_ptr<T, Deleter>(p, Deleter());
    }

    void Release()
    {
      buffer_resource.release();
    }

  private:
    template<class T>
    using Allocator = std::pmr::polymorphic_allocator<T>;

    template<class T>
    Allocator<T> GetAllocator()
    {
      return Allocator<T>{&buffer_resource};
    }

    // Compiler did not like my Lambda ...
    struct Deleter
    {
      template<class T>
      void operator()(T* p) { std::destroy_at(p); }
    };

  private:
    std::pmr::monotonic_buffer_resource buffer_resource;
};

} // namespace util
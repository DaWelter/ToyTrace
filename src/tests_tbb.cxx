#include <tbb/atomic.h>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb_thread.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_do.h>
#include <tbb/task_group.h>
#include <tbb/task.h>
#include <tbb/flow_graph.h>
#include <tbb/concurrent_hash_map.h>

#include <thread>
#include <iostream>
#include <optional>
#include <random>
#include <unordered_map>
#include <sstream>

#include "util_thread.hxx"

namespace util {
void Sleep(uint32_t ms)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}


static tbb::mutex cout_mutex;

// Nice debug output.
template<typename... Xs>
void printl(Xs...xs)
{
  tbb::mutex::scoped_lock lock(cout_mutex);
  printl_recurse(xs...);
}

template<typename X, typename... Xs>
void printl_recurse(X x, Xs...xs)
{
  std::cout << x;
  if constexpr (sizeof...(xs) > 0)
    printl_recurse(xs...);
  else
    std::cout << std::endl;
}

}

using util::printl;


namespace rendering_main_loop_parallelization1
{

/*  What's this? Sketch of the logic which could potentially be used in my path tracing algorithm.
 *
 * The problem is that I want to interrupt the rendering thread to prepare an image for display.
 *  The image preparation (here denoted secondary_task) should also be multithreaded.
 *  After the that is done, the primary tasks should continue.
 *
 *  Furthermore I want to run the tasks as much in order as possible. In terms of pixels to render,
 *  from left to right, top to bottom, for instance.
 */

class Demo
{
  tbb::atomic<bool> interrupt_flag = false;  // Interrupt the primary task.
  tbb::mutex mutex;  // Lock the frame buffer. Well here it actually locks std::cout.
  int count = 50;
  tbb::atomic<int> subtask_num = 0; // Iterate over pixels
public:
  static constexpr int n_tasks = 4;
  void Run()
  {
    while (!is_done())
    {
      run_primary_tasks();
      maybe_run_secondary_tasks();
    }
  }

  void run_primary_tasks()
  {
    // Rather than over the work items, the outer (parallel) 
    // iteration is used to launch worker tasks, each of which 
    // iterates over work items by itself.
    // This ensures the ordering I want.
    tbb::parallel_for(0, n_tasks, 1, [this](int worker_num) {
      int j;
      while (!interrupt_flag.load() && (j = subtask_num.fetch_and_increment()) < count)
      {
        printl("Running primary", worker_num, ", subtask ", j, ", threadid ", std::this_thread::get_id());
        util::Sleep(200);
      }
    });
  }

  void maybe_run_secondary_tasks()
  {
    if (interrupt_flag.load() || is_done())
    {
      tbb::parallel_for(0, n_tasks, 1, [this](int worker_num) {
        printl("Running secondary ", worker_num, ", threadid ", std::this_thread::get_id());
        util::Sleep(100);
      });
      interrupt_flag.store(false);
    }
  }

  void interrupt()
  {
    interrupt_flag.store(true);
  }

  bool is_done()
  {
    return subtask_num.load() > count;
  }

private:
};


// Same as above but slightly different implementation
class Demo2
{
  tbb::task_group other_kind_of_tg;
  tbb::mutex mutex;  // Lock the frame buffer. Well here it actually locks std::cout.
  int count = 50;
  tbb::atomic<int> subtask_num = 0; // Iterate over pixels
public:
  static constexpr int n_tasks = 4;
  void Run()
  {
    while_parallel_fed_interruptible(
      /* func = */[=](int j, int worker_num)
    {
      run_primary_task(j);
    },
      /*feeder = */ [=]() -> std::optional<int>
    {
      int j = subtask_num.fetch_and_increment();
      return (j >= count) ? std::optional<int>{} : j;
    },
      /* irq handler=*/ [=]() -> bool
    {
      run_secondary_tasks();
      return !is_done();
    },
      n_tasks, other_kind_of_tg);
  }

  void run_primary_task(int j)
  {
    printl("Running primary subtask ", j, ", threadid ", std::this_thread::get_id());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  void run_secondary_tasks()
  {
    tbb::parallel_for(0, n_tasks, 1, [this](int worker_num) {
      printl("Running secondary ", worker_num, ", threadid ", std::this_thread::get_id());
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    });
  }

  void interrupt()
  {
    other_kind_of_tg.cancel();
  }

  bool is_done()
  {
    return subtask_num.load() > count;
  }
};


class Demo3
{
    tbb::task_group task_group;
    tbb::task_arena arena;
    tbb::mutex mutex;  // Lock the frame buffer. Well here it actually locks std::cout.
    int count = 50;
public:
    static constexpr int n_tasks = 4;
    void Run()
    {
        arena.initialize(n_tasks);
        arena.execute([this] {
            parallel_for_interruptible(
                0, count, 1,
                /* func = */[=](int i)
            {
                run_primary_task(i);
            },
                /* irq handler=*/ [=]() -> bool
            {
                run_secondary_tasks();
                return true;
            }, task_group);
        });
    }

    void run_primary_task(int j)
    {
        printl("Running primary subtask ", j, ", threadid ", std::this_thread::get_id());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    void run_secondary_tasks()
    {
        tbb::parallel_for(0, n_tasks, 1, [this](int worker_num) {
            printl("Running secondary ", worker_num, ", threadid ", std::this_thread::get_id());
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        });
    }

    void interrupt()
    {
        task_group.cancel();
    }
};


void DoDemo()
{
  tbb::task_scheduler_init init(Demo::n_tasks); // Not strictly required. But this way I can set the number of threads.
  Demo3 d;
  // The demo algorithm must be run in a thread, so I can cause interruptions from the main thread.
  tbb::tbb_thread th([&] {d.Run(); });
  util::Sleep(200);
  d.interrupt();
  util::Sleep(500);
  d.interrupt();
  th.join();
}

}


namespace rendering_main_loop_with_flow
{

namespace tf = tbb::flow;

struct Item
{
  uint32_t frameId;
  uint32_t tileId;
};

class Accumulator
{
  std::unordered_map<uint32_t, uint32_t> num_received;
  int expected_count = 10;

public:
  Accumulator()
  {
  }

  void operator()(const Item &v)
  {
    num_received[v.frameId]++;
    if (num_received[v.frameId] == expected_count)
    {
      printl("I have received all stuff!");
    }
  }
};


class Demo
{
  std::mt19937 random_engine;
  tbb::spin_mutex random_mutex;

  const uint32_t num_items = 10;
  const uint32_t num_frames = 2;
  uint32_t counter = 0;
  uint32_t current_frame_id = 0;

  tbb::spin_rw_mutex fbmutex;
  std::vector<double> framebuffer;
  std::vector<int> image;

public:
  Demo()
    : framebuffer(num_items, 0.),
      image(num_items, -1)
  {
  }

  int SafeRandomInt(int a, int b)
  {
    decltype(random_mutex)::scoped_lock lock(random_mutex);
    return std::uniform_int_distribution(a, b)(random_engine);
  }

  void RunImageGeneration()
  {
    printl("start generating image");
    util::Sleep(50);
    std::vector<double> fbcopy;
    {
      decltype(fbmutex)::scoped_lock lock(fbmutex, /*write*/ true);
      fbcopy = this->framebuffer;
    }
    {std::stringstream ss;
    ss << "[";
    for (const auto x : fbcopy)
      ss << x << ", ";
    ss << "]";
    printl("FB copy = ", ss.str()); }

    tbb::parallel_for<size_t>(0, fbcopy.size(), [&](std::size_t i) {
      printl("Parallel image gen section ", i);
      image[i] = fbcopy[i];
      util::Sleep(SafeRandomInt(20, 25));
    });

    printl("Image Display");
    util::Sleep(600);
    {std::stringstream ss;
    ss << "[";
    for (const auto x : image)
      ss << x << ", ";
    ss << "]";
    printl("Image = ", ss.str()); }
  }

  void Run()
  {
    tf::graph graph;

    tf::source_node<Item> source(graph, [&](Item &out) {
       printl("gen ", this->current_frame_id, ", ", this->counter);
      if (counter < num_items)
      {
        out = { current_frame_id, counter++ };
        return true;
      }
      else
        return false;
      // Note: output argument is discarded if return value is false
    }, false);

    tf::function_node<Item, Item, tf::rejecting> process(graph, 3, [&](const Item &v) {
      uint32_t wait_time = 200;
      
      wait_time += SafeRandomInt(0, 20);

      printl("beginning work on ", v.tileId);
      util::Sleep(wait_time);
      {
        decltype(fbmutex)::scoped_lock lock(fbmutex, /*write=*/ false);
        framebuffer[v.tileId]++;
      }
      printl("finished work on ", v.tileId);
      return v;
    });

    tf::function_node<Item> accum(graph, 0, Accumulator());


    tf::graph image_gen_graph;

    tf::limiter_node<tf::continue_msg> update_image_trigger(image_gen_graph, 1);

    tf::continue_node<tf::continue_msg, tf::rejecting> update_image(image_gen_graph, [&](const tf::continue_msg &v) {
      this->RunImageGeneration();
      return tf::continue_msg{};
    });

    tf::make_edge(update_image_trigger, update_image);
    tf::make_edge(update_image, update_image_trigger.decrement);

    tf::make_edge(source, process);
    tf::make_edge(process, accum);

    for (current_frame_id = 0; current_frame_id < 2; ++current_frame_id)
    {
      printl("----- ROUND ", current_frame_id, " -----");
      counter = 0;
      source.activate();

      util::Sleep(100);
      printl("Request image gen = ", update_image_trigger.try_put(tf::continue_msg{}));
      

      util::Sleep(100);
      printl("Request image gen =", update_image_trigger.try_put(tf::continue_msg{}));

      graph.wait_for_all();
    }
    printl("Request image gen = ", update_image_trigger.try_put(tf::continue_msg{}));
    image_gen_graph.wait_for_all();
  }
};





}



int main(int argc, char **argv)
{
  rendering_main_loop_parallelization1::DoDemo();
  //rendering_main_loop_with_flow::Demo().Run();
  return 0;
}

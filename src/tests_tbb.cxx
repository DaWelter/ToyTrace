#include <tbb/atomic.h>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb_thread.h>
#include <tbb/task_scheduler_init.h>
#include <thread>
#include <iostream>

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
    tbb::parallel_for(0, n_tasks, 1, [this](int worker_num){
      int j;
      while (!interrupt_flag.load() && (j = subtask_num.fetch_and_increment())<count)
      {
        this->printl("Running primary",worker_num,", subtask ", j, ", threadid ", std::this_thread::get_id());
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
    });
  }
  
  void maybe_run_secondary_tasks()
  {
    if (interrupt_flag.load() || is_done())
    {
      tbb::parallel_for(0, n_tasks, 1, [this](int worker_num){
        this->printl("Running secondary ", worker_num, ", threadid ", std::this_thread::get_id());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
  // Nice debug output.
  template<typename... Xs>
  void printl(Xs...xs)
  {
    tbb::mutex::scoped_lock lock(mutex);
    printl_recurse(xs...);
  }
  
  template<typename X, typename... Xs>
  void printl_recurse(X x, Xs...xs)
  {
    std::cout << x;
    if constexpr(sizeof...(xs) > 0)
      printl_recurse(xs...);
    else
      std::cout << std::endl;
  }
};


int main(int argc, char **argv)
{
  tbb::task_scheduler_init init(Demo::n_tasks); // Not strictly required. But this way I can set the number of threads.
  Demo d;
  // The demo algorithm must be run in a thread, so I can cause interruptions from the main thread.
  tbb::tbb_thread th([&]{d.Run();}); 
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  d.interrupt();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  d.interrupt();
  th.join();
}

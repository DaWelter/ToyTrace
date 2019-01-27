#include <tbb/atomic.h>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb_thread.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_do.h>
#include <tbb/task_group.h>
#include <tbb/task.h>
#include <thread>
#include <iostream>
#include <boost/optional/optional.hpp>
#include <boost/none.hpp>

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


namespace ThreadUtilDetail
{

template<class Func, class Feeder>
struct while_parallel_fed_
{
  Func f;
  Feeder feeder;
  tbb::task_group *tg;
  int n_tasks;
  
  bool run()
  {
    for (int worker_num=0; worker_num<n_tasks; ++worker_num)
    {
      tg->run([=]{parallel_op();});
    }
    return tg->wait() == tbb::task_group_status::complete;
  }
  
  void parallel_op()
  {
    auto val = feeder();
    if (!val)
      return;
    f(*val);
    tg->run([=]{parallel_op();});
  }
};


template<class Func, class Feeder>
bool while_parallel_fed(Func &&f, Feeder &&feeder, int n_tasks, tbb::task_group &tg)
{
  return while_parallel_fed_<Func,Feeder>{f, feeder, &tg, n_tasks}.run();
}


template<class Func, class Feeder, class IrqHandler>
void while_parallel_fed_interruptible(Func &&f, Feeder &&feeder, IrqHandler &&irq_handler, int n_tasks, tbb::task_group &tg)
{
  while(true)
  {
    bool completed = while_parallel_fed(std::move(f), std::move(feeder), n_tasks, tg);
    if (completed || !irq_handler())
      break;
  }
}

}

using ThreadUtilDetail::while_parallel_fed;
using ThreadUtilDetail::while_parallel_fed_interruptible;


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
      /* func = */[=](int j)
      {
        run_primary_task(j);
      }, 
      /*feeder = */ [=]() -> boost::optional<int> 
      {
        int j = subtask_num.fetch_and_increment();
        return (j>=count) ? boost::optional<int>{} : j;
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
    this->printl("Running primary subtask ", j, ", threadid ", std::this_thread::get_id());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  
  void run_secondary_tasks()
  {
    tbb::parallel_for(0, n_tasks, 1, [this](int worker_num){
      this->printl("Running secondary ", worker_num, ", threadid ", std::this_thread::get_id());
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


void DoDemo()
{
  tbb::task_scheduler_init init(Demo::n_tasks); // Not strictly required. But this way I can set the number of threads.
  Demo2 d;
  // The demo algorithm must be run in a thread, so I can cause interruptions from the main thread.
  tbb::tbb_thread th([&]{d.Run();}); 
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  d.interrupt();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  d.interrupt();
  th.join();  
}


int main(int argc, char **argv)
{
  DoDemo();
  return 0;
}

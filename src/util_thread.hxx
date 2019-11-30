#pragma once

#include <tbb/parallel_do.h>
#include <tbb/task_group.h>


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
      tg->run([=]{parallel_op(worker_num);});
    }
    return tg->wait() == tbb::task_group_status::complete;
  }
  
  void parallel_op(int worker_num)
  {
    auto val = feeder();
    if (!val)
      return;
    f(*val, worker_num);
    tg->run([=]{parallel_op(worker_num);});
  }
};

/* Runs f in parallel using n_tasks number of tasks.
 * The items processed by f are produced by feeder. 
 * The feeder function should return an optional with the work item, or an empty optional when all items were processed.
 */
template<class Func, class Feeder>
bool while_parallel_fed(Func &&f, Feeder &&feeder, int n_tasks, tbb::task_group &tg)
{
  return while_parallel_fed_<Func,Feeder>{f, feeder, &tg, n_tasks}.run();
}


/* Like while_parallel_fed, but has the ability to interrupt the work, invoke irq_handler and resume.
 * The way this works is, that you call tg.cancel(), then after all in-flight work items were processed,
 * you get an invocation of irq_handler.
 * irq_handler should return true if work should resume. Else while_parallel_fed_interruptible will return
 * without processing more items.
 * Example: see tests_tbb.cxx
 */
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


template<class Func, class IrqHandler>
void parallel_for_interruptible(int start, int end, int step, Func &&f, IrqHandler &&irq_handler, tbb::task_group &tg)
{
    const int workers = tbb::this_task_arena::max_concurrency();
    tbb::atomic<int> idx{ start };

    std::function<void()> worker_fun;
    
    worker_fun = [&idx, &tg, step, end, &f, &worker_fun]()
    {
        int my_idx = idx.fetch_and_add(step);
        if (my_idx < end)
        {
            f(my_idx);
            tg.run(worker_fun);
        }
    };

    while (true)
    {
        for (int i=0; i<workers; ++i)
        {
            tg.run(worker_fun);
        }
        if (tg.wait() == tbb::task_group_status::complete || !irq_handler())
            break;
    }
}


} // namespace ThreadUtilDetail


using ThreadUtilDetail::while_parallel_fed;
using ThreadUtilDetail::while_parallel_fed_interruptible;
using ThreadUtilDetail::parallel_for_interruptible;
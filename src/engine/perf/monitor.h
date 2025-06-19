

#ifndef OURS_MONITOR_H
#define OURS_MONITOR_H

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "OURSdata/dataset/engine/perf/profiling.h"
#include "OURSdata/dataset/util/cond_var.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class ExecutionTree;
class ProfilingManager;
class ConfigManager;

class Monitor {
 public:
  // Monitor object constructor
  explicit Monitor(ProfilingManager *profiling_manager);

  ~Monitor();

  // Functor for Perf Monitor main loop.
  // This function will be the entry point of our::Dataset::Task
  Status operator()();

  // Setter for execution tree pointer
  void SetTree(ExecutionTree *tree) { tree_ = tree; }

 private:
  // private constructor
  Monitor(ProfilingManager *profiling_manager, const std::shared_ptr<ConfigManager> &cfg);

  ProfilingManager *profiling_manager_;
  int64_t sampling_interval_;
  ExecutionTree *tree_;
  std::mutex mux_;
  CondVar cv_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_MONITOR_H

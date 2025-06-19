
#include "OURSdata/dataset/engine/perf/monitor.h"
#include "OURSdata/dataset/core/config_manager.h"
#include "OURSdata/dataset/engine/execution_tree.h"
#include "OURSdata/dataset/engine/consumers/tree_consumer.h"

namespace ours {
namespace dataset {

Monitor::Monitor(ProfilingManager *profiling_manager) : Monitor(profiling_manager, GlobalContext::config_manager()) {}

Monitor::Monitor(ProfilingManager *profiling_manager, const std::shared_ptr<ConfigManager> &cfg)
    : profiling_manager_(profiling_manager), sampling_interval_(cfg->monitor_sampling_interval()) {
  if (profiling_manager_ != nullptr) {
    tree_ = profiling_manager_->tree_;
  }
}

Monitor::~Monitor() {
  // just set the pointer to nullptr, it's not be released here
  if (profiling_manager_) {
    profiling_manager_ = nullptr;
  }

  if (tree_) {
    tree_ = nullptr;
  }
}

Status Monitor::operator()() {
  // Register this thread with TaskManager to receive proper interrupt signal.
  TaskManager::FindMe()->Post();
  std::unique_lock<std::mutex> _lock(mux_);

  // Keep sampling if
  // 1) Monitor Task is not interrupted by TaskManager AND
  // 2) Iterator has not received EOF

  while (!this_thread::is_interrupted() && !(tree_->isFinished())) {
    if (tree_->IsEpochEnd()) {
      tree_->SetExecuting();
    }
    for (auto &node : profiling_manager_->GetSamplingNodes()) {
      RETURN_IF_NOT_OK(node.second->Sample());
    }
    RETURN_IF_NOT_OK(cv_.WaitFor(&_lock, sampling_interval_));
  }
  MS_LOG(INFO) << "Monitor Thread terminating...";
  return Status::OK();
}

}  // namespace dataset
}  // namespace ours

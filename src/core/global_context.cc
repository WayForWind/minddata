
#include "OURSdata/dataset/core/global_context.h"

#include <memory>
#include <mutex>

#include "OURSdata/dataset/core/config_manager.h"
#include "OURSdata/dataset/core/cv_tensor.h"
#include "OURSdata/dataset/core/device_tensor.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/engine/perf/profiling.h"
#include "OURSdata/dataset/util/allocator.h"
#include "OURSdata/dataset/util/system_pool.h"

namespace ours {
namespace dataset {
// Global static pointer for the singleton GlobalContext
std::unique_ptr<GlobalContext> GlobalContext::global_context_ = nullptr;
std::once_flag GlobalContext::init_instance_flag_;

constexpr int GlobalContext::kArenaSize;
constexpr int GlobalContext::kMaxSize;
constexpr bool GlobalContext::kInitArena;

// Singleton initializer
GlobalContext *GlobalContext::Instance() {
  // If the single global context is not created yet, then create it. Otherwise the
  // existing one is returned.
  std::call_once(init_instance_flag_, []() {
    global_context_.reset(new GlobalContext());
    Status rc = global_context_->Init();
    if (rc.IsError()) {
      std::terminate();
    }
  });
  return global_context_.get();
}

Status GlobalContext::Init() {
  config_manager_ = std::make_shared<ConfigManager>();
  mem_pool_ = std::make_shared<SystemPool>();
  // For testing we can use Dummy pool instead

  // Create some tensor allocators for the different types and hook them into the pool.
  tensor_allocator_ = std::make_unique<Allocator<Tensor>>(mem_pool_);
#if defined(ENABLE_OURSdata_PYTHON)
  cv_tensor_allocator_ = std::make_unique<Allocator<CVTensor>>(mem_pool_);
#endif
  device_tensor_allocator_ = std::make_unique<Allocator<DeviceTensor>>(mem_pool_);
  int_allocator_ = std::make_unique<IntAlloc>(mem_pool_);
  profiler_manager_ = std::make_shared<ProfilingManager>();
  return Status::OK();
}

// A print method typically used for debugging
void GlobalContext::Print(std::ostream &out) const {
  out << "GlobalContext contains the following default config: " << *config_manager_ << "\n";
}
}  // namespace dataset
}  // namespace ours

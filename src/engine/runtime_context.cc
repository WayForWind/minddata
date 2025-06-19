

#include "OURSdata/dataset/engine/runtime_context.h"
namespace ours::dataset {
void RuntimeContext::AssignConsumer(std::shared_ptr<TreeConsumer> tree_consumer) {
  tree_consumer_ = std::move(tree_consumer);
}
Status NativeRuntimeContext::Terminate() {
  MS_LOG(INFO) << "Terminating a Dataset NativeRuntime.";
  if (tree_consumer_ != nullptr) {
    return TerminateImpl();
  }
  MS_LOG(INFO) << "Dataset TreeConsumer was not initialized.";
  return Status::OK();
}

Status NativeRuntimeContext::TerminateImpl() {
  CHECK_FAIL_RETURN_UNEXPECTED(tree_consumer_ != nullptr, "Dataset TreeConsumer is not initialized.");
  return tree_consumer_->Terminate();
}

NativeRuntimeContext::~NativeRuntimeContext() {
  Status rc = NativeRuntimeContext::Terminate();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Error while terminating the consumer. Message:" << rc;
  }
}

TreeConsumer *RuntimeContext::GetConsumer() { return tree_consumer_.get(); }

Status RuntimeContext::Init() const { return GlobalInit(); }
}  // namespace ours::dataset

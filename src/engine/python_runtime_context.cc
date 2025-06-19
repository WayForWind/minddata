

#include "OURSdata/dataset/engine/python_runtime_context.h"
#include "pybind11/pybind11.h"

namespace ours::dataset {
Status PythonRuntimeContext::Terminate() {
  MS_LOG(INFO) << "Terminating a Dataset PythonRuntime.";
  if (tree_consumer_ != nullptr) {
    return TerminateImpl();
  }
  MS_LOG(INFO) << "Dataset TreeConsumer was not initialized.";
  return Status::OK();
}

Status PythonRuntimeContext::TerminateImpl() {
  CHECK_FAIL_RETURN_UNEXPECTED(tree_consumer_ != nullptr, "Dataset TreeConsumer is not initialized.");
  // Release GIL before joining all threads
  py::gil_scoped_release gil_release;
  return tree_consumer_->Terminate();
}

PythonRuntimeContext::~PythonRuntimeContext() {
  Status rc = PythonRuntimeContext::Terminate();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Error while terminating the consumer. Message:" << rc;
  }
  if (tree_consumer_) {
    tree_consumer_.reset();
  }
}

TreeConsumer *PythonRuntimeContext::GetPythonConsumer() {
  if (GlobalContext::config_manager()->get_debug_mode()) {
    return dynamic_cast<PythonPullBasedIteratorConsumer *>(tree_consumer_.get());
  } else {
    return dynamic_cast<PythonIteratorConsumer *>(tree_consumer_.get());
  }
}
}  // namespace ours::dataset

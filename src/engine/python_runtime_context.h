
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_PYTHON_RUNTIME_CONTEXT_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_PYTHON_RUNTIME_CONTEXT_H_

#include "OURSdata/dataset/core/client.h"
#include "OURSdata/dataset/engine/consumers/tree_consumer.h"
#include "OURSdata/dataset/engine/consumers/python_tree_consumer.h"
#include "OURSdata/dataset/engine/runtime_context.h"

namespace ours::dataset {
class NativeRuntimeContext;

/// Class that represents Python single runtime instance which can consume data from a data pipeline
class PythonRuntimeContext : public RuntimeContext {
 public:
  /// Method to terminate the runtime, this will not release the resources
  /// \return Status error code
  Status Terminate() override;

  /// Safe destructing the tree that includes python objects
  ~PythonRuntimeContext() override;

  TreeConsumer *GetPythonConsumer();

 private:
  /// Internal function to perform the termination
  /// \return Status error code
  Status TerminateImpl();
};

}  // namespace ours::dataset
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_PYTHON_RUNTIME_CONTEXT_H_

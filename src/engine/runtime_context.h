
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_RUNTIME_CONTEXT_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_RUNTIME_CONTEXT_H_

#include <memory>
#include "OURSdata/dataset/core/client.h"
#include "OURSdata/dataset/engine/consumers/tree_consumer.h"

namespace ours::dataset {
class TreeConsumer;
/// Class that represents single runtime instance which can consume data from a data pipeline
class RuntimeContext {
 public:
  /// Default constructor
  RuntimeContext() = default;

  /// Initialize the runtime, for now we just call the global init
  /// \return Status error code
  Status Init() const;

  /// Set the tree consumer
  /// \param tree_consumer to be assigned
  void AssignConsumer(std::shared_ptr<TreeConsumer> tree_consumer);

  /// Get the tree consumer
  /// \return Raw pointer to the tree consumer.
  TreeConsumer *GetConsumer();

  /// Method to terminate the runtime, this will not release the resources
  /// \return Status error code
  virtual Status Terminate() = 0;

  virtual ~RuntimeContext() = default;

  std::shared_ptr<TreeConsumer> tree_consumer_;
};

/// Class that represents C++ single runtime instance which can consume data from a data pipeline
class NativeRuntimeContext : public RuntimeContext {
 public:
  /// Method to terminate the runtime, this will not release the resources
  /// \return Status error code
  Status Terminate() override;

  ~NativeRuntimeContext() override;

 private:
  /// Internal function to perform the termination
  /// \return Status error code
  Status TerminateImpl();
};

}  // namespace ours::dataset
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_RUNTIME_CONTEXT_H_

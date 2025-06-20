

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_TREE_ADAPTER_LITE_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_TREE_ADAPTER_LITE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "OURSdata/dataset/engine/ir/datasetops/dataset_node.h"

namespace ours {
namespace dataset {

class TensorRow;
class DatasetNode;

class TreeAdapterLite {
 public:
  // this flag is used to indicate the purpose of the creation of this tree adapter (type of the tree_consumer).
  // Currently there are 3 types of consumer, Iterator, Getter and TDT/Vocab/Save ...
  // To avoid premature optimization, the last type (TDT/Vocab/Save) is regarded as Iterator for now.
  enum UsageFlag { kDeIterator = 0, kDeGetter = 1, kDeReset = 2 };

  explicit TreeAdapterLite(UsageFlag usage = kDeGetter);

  ~TreeAdapterLite() = default;

  Status BuildTree(std::shared_ptr<DatasetNode> root_ir);

  // Get rows equal to num_rows
  Status GetNextRow(TensorRow *const row);

  std::unordered_map<std::string, int32_t> GetColumnNameMap() const { return tree_->root()->column_name_id_map(); }

  // unique_ptr overloads operator bool(), will return false if it doesn't manage an object
  std::weak_ptr<DatasetOp> GetRoot() const { return tree_ ? tree_->root() : nullptr; }

  // This function performs syntax checking, semantics checking, and then call BuildTree
  Status Compile(const std::shared_ptr<DatasetNode> &input_ir, int32_t num_epochs = -1);

 protected:
  // Run the mandatory pass checking the syntax and semantics of the IR tree
  Status PrePass(std::shared_ptr<DatasetNode> ir);

  // Run the mandatory pass augmenting the IR tree
  Status PostPass(std::shared_ptr<DatasetNode> ir) const;

  // Return Offload Json
  nlohmann::json GetOffloadJson();

  std::shared_ptr<DatasetNode> input_ir_;
  std::shared_ptr<DatasetNode> root_ir_;

 private:
  // This RECURSIVE function walks the (optimized) IR tree in DFS to build its corresponding Execution tree.
  Status BuildExecutionTreeRecur(std::shared_ptr<DatasetNode> ir, std::shared_ptr<DatasetOp> *op);

  std::shared_ptr<DatasetOp> root_;  // current connector capacity of root op, used for profiling
  std::unique_ptr<ExecutionTree> tree_;
  UsageFlag usage_;  // usage of this tree adapter (type of consumer)
  nlohmann::json offload_json_;
};

}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_TREE_ADAPTER_LITE_H_



#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_PRE_NODE_REMOVAL_PASS_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_PRE_NODE_REMOVAL_PASS_H_

#include <memory>
#include <vector>
#include "OURSdata/dataset/engine/opt/pass.h"

namespace ours {
namespace dataset {

class DatasetOp;

/// \class RemovalPass removal_pass.h
/// \brief This is a tree pass that will remove nodes.  It uses removal_nodes to first identify which
///     nodes should be removed, and then removes them.
class NodeRemovalPass : public IRTreePass {
  /// \class RemovalNodes
  /// \brief This is a NodePass whose job is to identify which nodes should be removed.
  ///     It works in conjunction with the removal_pass.
  class RemovalNodes : public IRNodePass {
   public:
    /// \brief Constructor
    RemovalNodes();

    /// \brief Destructor
    ~RemovalNodes() = default;

    /// \brief Perform RepeatNode removal check
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<RepeatNode> node, bool *const modified) override;

    /// \brief Perform SkipNode removal check
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<SkipNode> node, bool *const modified) override;

    /// \brief Perform TakeNode removal check
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<TakeNode> node, bool *const modified) override;

    /// \brief Getter
    /// \return All the nodes to be removed
    std::vector<std::shared_ptr<DatasetNode>> nodes_to_remove() { return nodes_to_remove_; }

   private:
    std::vector<std::shared_ptr<DatasetNode>> nodes_to_remove_;
  };

 public:
  /// \brief Constructor
  NodeRemovalPass();

  /// \brief Destructor
  ~NodeRemovalPass() = default;

  /// \brief Runs a removal_nodes pass first to find out which nodes to remove, then removes them.
  /// \param[in, out] tree The tree to operate on.
  /// \param[in, out] Indicate of the tree was modified.
  /// \return Status The status code returned
  Status RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) override;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_PRE_NODE_REMOVAL_PASS_H_

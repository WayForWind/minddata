

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_POST_GENERATOR_NODE_PASS_H
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_POST_GENERATOR_NODE_PASS_H

#include <memory>
#include <utility>
#include <vector>
#include "OURSdata/dataset/engine/opt/pass.h"

namespace ours {
namespace dataset {

/// \class GeneratorNodePass repeat_pass.h
/// \brief This is a NodePass who's job is to perform setup actions for RepeatOps. A RepeatOp needs to have references
///     to the eoe-producing (typically leaf) nodes underneath it.
class GeneratorNodePass : public IRNodePass {
 public:
  /// \brief Constructor
  GeneratorNodePass();

  /// \brief Destructor
  ~GeneratorNodePass() = default;

  /// \brief Record the starting point to collect the Generator node
  /// \param[in] node The node being visited
  /// \param[in, out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<RepeatNode> node, bool *const modified) override;

  /// \brief Record the starting point to collect the Generator node
  /// \param[in] node The node being visited
  /// \param[in, out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<EpochCtrlNode> node, bool *const modified) override;

  /// \brief Add the Generator node to the set
  /// \param[in] node The node being visited
  /// \param[in, out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<GeneratorNode> node, bool *const modified) override;

  /// \brief Add the Generator node(s) from the set to this Repeat node for run-time processing
  /// \param[in] node The node being visited
  /// \param[in, out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<RepeatNode> node, bool *const modified) override;

  /// \brief Add the Generator node(s) from the set to this EpochCtrl node for run-time processing
  /// \param[in] node The node being visited
  /// \param[in, out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<EpochCtrlNode> node, bool *const modified) override;

 private:
  std::vector<std::shared_ptr<RepeatNode>> repeat_ancestors_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_POST_GENERATOR_NODE_PASS_H



#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_PASS_POST_REPEAT_PASS_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_PASS_POST_REPEAT_PASS_

#include <memory>
#include <stack>
#include <utility>
#include "OURSdata/dataset/engine/opt/pass.h"

namespace ours {
namespace dataset {

/// \class RepeatPass
/// \brief This is a post pass that calculate the number of repeats the pipeline needs to fetch the data.
class RepeatPass : public IRNodePass {
 public:
  using op_stack = std::stack<std::shared_ptr<DatasetNode>>;

  /// \brief Constructor
  RepeatPass();

  /// \brief Destructor
  ~RepeatPass() = default;

  /// \brief Identifies the subtree below this node as being in a repeated path of the tree.
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<RepeatNode> node, bool *const modified) override;

  /// \brief Identifies the subtree below this node as being in a repeated path of the tree.
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<EpochCtrlNode> node, bool *const modified) override;


  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned


  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned


  /// \brief Hooks up any identified eoe nodes under this repeat.
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<RepeatNode> node, bool *const modified) override;

  /// \brief Hooks up any identified eoe nodes under this repeat.
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<EpochCtrlNode> node, bool *const modified) override;


  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned


  /// \brief Turns off the tracking for operations under merge op
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned


  /// \brief Saves the lookup up in case it needs to be referenced by a repeat
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned


  /// \brief Sets the epoch count for DataQueueNode
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<DataQueueNode> node, bool *const modified) override;

  /// \brief All operators have a flag that might be set related to the repeat and any leaf nodes need to be set up
  ///     for use with a controlling repeat above it.
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<DatasetNode> node, bool *const modified) override;

 private:





  int32_t num_repeats_;                        // A multiplier to the total number of repeats
  int32_t num_epochs_;                         // To save the total number of epochs


};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_PASS_POST_REPEAT_PASS_

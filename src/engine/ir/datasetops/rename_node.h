

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_RENAME_NODE_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_RENAME_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/engine/ir/datasetops/dataset_node.h"

namespace ours {
namespace dataset {
class RenameNode : public DatasetNode {
 public:
  /// \brief Constructor
  explicit RenameNode(std::shared_ptr<DatasetNode> child, const std::vector<std::string> &input_columns,
                      const std::vector<std::string> &output_columns);

  /// \brief Destructor
  ~RenameNode() override = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kRenameNode; }

  /// \brief Base-class override for accepting IRNodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status Accept(IRNodePass *const p, bool *const modified) override;

  /// \brief Base-class override for accepting IRNodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status AcceptAfter(IRNodePass *const p, bool *const modified) override;

  /// \brief Print the description
  /// \param out - The output stream to write output to
  void Print(std::ostream &out) const override;

  /// \brief Copy the node to a new object
  /// \return A shared pointer to the new copy
  std::shared_ptr<DatasetNode> Copy() override;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \param node_ops - A vector containing shared pointer to the Dataset Ops that this object will create
  /// \return Status Status::OK() if build successfully
  Status Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

  /// \brief Getter functions
  const std::vector<std::string> &InputColumns() const { return input_columns_; }
  const std::vector<std::string> &OutputColumns() const { return output_columns_; }

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

  /// \brief Function for read dataset operation from json
  /// \param[in] json_obj The JSON object to be deserialized
  /// \param[in] ds dataset node constructed
  /// \param[out] result Deserialized dataset after the operation
  /// \return Status The status code returned
  static Status from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> ds,
                          std::shared_ptr<DatasetNode> *result);

 private:
  std::vector<std::string> input_columns_;
  std::vector<std::string> output_columns_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_RENAME_NODE_H_

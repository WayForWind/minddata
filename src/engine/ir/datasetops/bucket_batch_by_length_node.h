

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_BUCKET_BATCH_BY_LENGTH_NODE_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_BUCKET_BATCH_BY_LENGTH_NODE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/engine/ir/datasetops/dataset_node.h"

namespace ours {
namespace dataset {
class BucketBatchByLengthNode : public DatasetNode {
 public:
  /// \brief Constructor
  BucketBatchByLengthNode(std::shared_ptr<DatasetNode> child, const std::vector<std::string> &column_names,
                          const std::vector<int32_t> &bucket_boundaries, const std::vector<int32_t> &bucket_batch_sizes,
                          std::shared_ptr<TensorOp> element_length_function = nullptr,
                          const std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> &pad_info = {},
                          bool pad_to_bucket_boundary = false, bool drop_remainder = false);

  /// \brief Destructor
  ~BucketBatchByLengthNode() override = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kBucketBatchByLengthNode; }

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

  bool IsSizeDefined() override { return false; };

  /// \brief Getter functions
  const std::vector<std::string> &ColumnNames() const { return column_names_; }
  const std::vector<int32_t> &BucketBoundaries() const { return bucket_boundaries_; }
  const std::vector<int32_t> &BucketBatchSizes() const { return bucket_batch_sizes_; }
  const std::shared_ptr<TensorOp> &ElementLengthFunction() const { return element_length_function_; }
  const std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> &PadInfo() const { return pad_info_; }
  bool PadToBucketBoundary() const { return pad_to_bucket_boundary_; }
  bool DropRemainder() const { return drop_remainder_; }

 private:
  std::vector<std::string> column_names_;
  std::vector<int32_t> bucket_boundaries_;
  std::vector<int32_t> bucket_batch_sizes_;
  std::shared_ptr<TensorOp> element_length_function_;
  std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> pad_info_;
  bool pad_to_bucket_boundary_;
  bool drop_remainder_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_BUCKET_BATCH_BY_LENGTH_NODE_H_

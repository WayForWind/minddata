/

#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_ADD_TOKEN_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_ADD_TOKEN_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class AddTokenOp : public TensorOp {
 public:
  /// \brief Constructor.
  /// \param[in] token The token to be added.
  /// \param[in] begin Whether to insert token at start or end of sequence.
  AddTokenOp(const std::string &token, bool begin) : token_(token), begin_(begin) {}

  ~AddTokenOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kAddTokenOp; }

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

 private:
  const std::string token_;
  bool begin_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_ADD_TOKEN_OP_H_

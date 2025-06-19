

#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_TO_VECTORS_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_TO_VECTORS_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/text/vectors.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class ToVectorsOp : public TensorOp {
 public:
  /// \brief Constructor.
  /// \param[in] vectors Vectors used to lookup tokens.
  /// \param[in] unk_init Vector used to initialize OOV token.
  /// \param[in] lower_case_backup Whether to look up the token in the lower case.
  ToVectorsOp(const std::shared_ptr<Vectors> &vectors, const std::vector<float> &unk_init, bool lower_case_backup);

  /// \brief Destructor.
  ~ToVectorsOp() = default;

  /// \brief Perform actual ToVectors on each tensor.
  /// \param[in] input Input tensor.
  /// \param[in] output Output tensor.
  /// \return[out] Status code.
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  /// \param[in] inputs DataType of input tensor.
  /// \param[in] outputs DataType of output tensor.
  /// \return[out] Status code.
  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  /// \brief Get Op name.
  std::string Name() const override { return kToVectorsOp; }

 private:
  std::shared_ptr<Vectors> vectors_;
  std::vector<float> unk_init_;
  bool lower_case_backup_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_TO_VECTORS_OP_H_

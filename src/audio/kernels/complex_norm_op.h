
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_COMPLEX_NORM_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_COMPLEX_NORM_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class ComplexNormOp : public TensorOp {
 public:
  /// \brief Constructor for ComplexNormOp.
  /// \param[in] power Power of the norm (Optional).
  explicit ComplexNormOp(float power = 1.0);

  ~ComplexNormOp() override = default;

  void Print(std::ostream &out) const override;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kComplexNormOp; }

 private:
  float power_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_COMPLEX_NORM_OP_H_

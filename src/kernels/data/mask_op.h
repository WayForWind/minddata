
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_MASK_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_MASK_OP_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"
#include "OURSdata/dataset/kernels/data/type_cast_op.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class MaskOp : public TensorOp {
 public:
  MaskOp(RelationalOp op, std::shared_ptr<Tensor> value, DataType type = DataType(DataType::DE_BOOL))
      : op_(op), value_(std::move(value)), type_(type), cast_(new TypeCastOp(type)) {}

  ~MaskOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kMaskOp; }

 private:
  RelationalOp op_;
  std::shared_ptr<Tensor> value_;
  DataType type_;
  std::unique_ptr<TypeCastOp> cast_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_MASK_OP_H_

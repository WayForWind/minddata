
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_TYPE_CAST_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_TYPE_CAST_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class TypeCastOp : public TensorOp {
 public:
  // Constructor for TypecastOp
  // @param data_type datatype to cast to
  explicit TypeCastOp(const DataType &data_type);

  // Constructor for TypecastOp
  // @param data_type datatype to cast to
  explicit TypeCastOp(const std::string &data_type);

  ~TypeCastOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kTypeCastOp; }

 private:
  DataType type_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_TYPE_CAST_OP_H_

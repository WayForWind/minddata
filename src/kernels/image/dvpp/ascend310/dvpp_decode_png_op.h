

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_PNG_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_PNG_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/data_type.h"
#include "OURSdata/dataset/core/device_resource.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/dvpp/acl_adapter.h"
#include "OURSdata/dataset/kernels/image/dvpp/utils/ErrorCode.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/log_adapter.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class DvppDecodePngOp : public TensorOp {
 public:
  DvppDecodePngOp() {}

  /// \brief Destructor
  ~DvppDecodePngOp() = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status Compute(const std::shared_ptr<DeviceTensor> &input, std::shared_ptr<DeviceTensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kDvppDecodePngOp; }

  Status SetAscendResource(const std::shared_ptr<DeviceResource> &resource) override;

 private:
  std::shared_ptr<void> processor_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_PNG_OP_H_

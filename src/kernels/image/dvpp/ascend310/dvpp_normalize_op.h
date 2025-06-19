

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_DVPP_NORMALIZE_JPEG_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_DVPP_NORMALIZE_JPEG_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/device_resource.h"
#include "OURSdata/dataset/core/device_tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/log_adapter.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class DvppNormalizeOp : public TensorOp {
 public:
  DvppNormalizeOp(std::vector<float> mean, std::vector<float> std) : mean_(std::move(mean)), std_(std::move(std)) {}

  ~DvppNormalizeOp() = default;

  Status Compute(const std::shared_ptr<DeviceTensor> &input, std::shared_ptr<DeviceTensor> *output) override;

  std::string Name() const override { return kDvppNormalizeOp; }

  Status SetAscendResource(const std::shared_ptr<DeviceResource> &resource) override;

 private:
  std::vector<float> mean_;
  std::vector<float> std_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_DVPP_NORMALIZE_JPEG_OP_H_

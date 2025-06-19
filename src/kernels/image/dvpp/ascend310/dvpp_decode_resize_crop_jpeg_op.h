

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_RESIZE_CROP_JPEG_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_RESIZE_CROP_JPEG_OP_H_

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
class DvppDecodeResizeCropJpegOp : public TensorOp {
 public:
  DvppDecodeResizeCropJpegOp(int32_t crop_height, int32_t crop_width, int32_t resized_height, int32_t resized_width)
      : crop_height_(crop_height),
        crop_width_(crop_width),
        resized_height_(resized_height),
        resized_width_(resized_width) {}

  /// \brief Destructor
  ~DvppDecodeResizeCropJpegOp() = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status Compute(const std::shared_ptr<DeviceTensor> &input, std::shared_ptr<DeviceTensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kDvppDecodeResizeCropJpegOp; }

  Status SetAscendResource(const std::shared_ptr<DeviceResource> &resource) override;

 private:
  int32_t crop_height_;
  int32_t crop_width_;
  int32_t resized_height_;
  int32_t resized_width_;
  std::shared_ptr<void> processor_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_RESIZE_CROP_JPEG_OP_H_

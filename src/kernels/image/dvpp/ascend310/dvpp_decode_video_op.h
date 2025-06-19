

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_VIDEO_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_VIDEO_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/data_type.h"
#include "OURSdata/dataset/core/device_resource.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/dvpp/acl_adapter.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"
#include "utils/log_adapter.h"

namespace ours {
namespace dataset {
class DvppDecodeVideoOp : public TensorOp {
 public:
  // Default values
  static const VdecOutputFormat kDefVdecOutputFormat;
  static const char kDefOutput[];

  DvppDecodeVideoOp(uint32_t width, uint32_t height, VdecStreamFormat type,
                    VdecOutputFormat out_format = kDefVdecOutputFormat, const std::string &output = kDefOutput)
      : width_(width), height_(height), format_(out_format), en_type_(type), output_(output) {}

  /// \brief Destructor
  ~DvppDecodeVideoOp() = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kDvppDecodeVideoOp; }

 private:
  uint32_t width_;
  uint32_t height_;

  /* 1：YUV420 semi-planner（nv12）
     2：YVU420 semi-planner（nv21）
  */
  VdecOutputFormat format_;

  /* 0：H265 main level
   * 1：H264 baseline level
   * 2：H264 main level
   * 3：H264 high level
   */
  VdecStreamFormat en_type_;
  std::string output_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_DVPP_DECODE_VIDEO_OP_H_



#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RANDOM_CROP_DECODE_RESIZE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RANDOM_CROP_DECODE_RESIZE_IR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"
#include "OURSdata/dataset/kernels/ir/vision/random_resized_crop_ir.h"

namespace ours {
namespace dataset {
namespace vision {
constexpr char kRandomCropDecodeResizeOperation[] = "RandomCropDecodeResize";

class RandomCropDecodeResizeOperation : public RandomResizedCropOperation {
 public:
  RandomCropDecodeResizeOperation(const std::vector<int32_t> &size, const std::vector<float> &scale,
                                  const std::vector<float> &ratio, InterpolationMode interpolation,
                                  int32_t max_attempts);

  explicit RandomCropDecodeResizeOperation(const RandomResizedCropOperation &base);

  ~RandomCropDecodeResizeOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RANDOM_CROP_DECODE_RESIZE_IR_H_

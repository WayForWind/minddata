

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RANDOM_RESIZED_CROP_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RANDOM_RESIZED_CROP_IR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace vision {
constexpr char kRandomResizedCropOperation[] = "RandomResizedCrop";

class RandomResizedCropOperation : public TensorOperation {
 public:
  RandomResizedCropOperation(const std::vector<int32_t> &size, const std::vector<float> &scale,
                             const std::vector<float> &ratio, InterpolationMode interpolation, int32_t max_attempts);

  /// \brief default copy constructor
  RandomResizedCropOperation(const RandomResizedCropOperation &);

  // Copy assignment operator
  RandomResizedCropOperation &operator=(const RandomResizedCropOperation &) = default;

  ~RandomResizedCropOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 protected:
  std::vector<int32_t> size_;
  std::vector<float> scale_;
  std::vector<float> ratio_;
  InterpolationMode interpolation_;
  int32_t max_attempts_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_RANDOM_RESIZED_CROP_IR_H_

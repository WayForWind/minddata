
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MU_LAW_ENCODING_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MU_LAW_ENCODING_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class MuLawEncodingOp : public TensorOp {
 public:
  /// \brief Constructor for MuLawEncoding.
  /// \param[in] quantization_channels Number of channels.
  explicit MuLawEncodingOp(int32_t quantization_channels = 256);

  ~MuLawEncodingOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kMuLawEncodingOp; }

 private:
  int32_t quantization_channels_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MU_LAW_ENCODING_OP_H_

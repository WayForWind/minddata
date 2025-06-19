
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_SLICE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_SLICE_OP_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/core/tensor_helpers.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class SliceOp : public TensorOp {
 public:
  explicit SliceOp(std::vector<SliceOption> slice_input) : slice_options_(std::move(slice_input)) {}

  explicit SliceOp(const SliceOption &slice_option) { slice_options_.push_back(slice_option); }

  // short hand notation for slicing along fist dimension
  explicit SliceOp(Slice slice) { (void)slice_options_.emplace_back(slice); }

  explicit SliceOp(bool all) { (void)slice_options_.emplace_back(all); }

  explicit SliceOp(const std::vector<dsize_t> &indices) { (void)slice_options_.emplace_back(indices); }

  ~SliceOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kSliceOp; }

 private:
  std::vector<SliceOption> slice_options_ = {};
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_SLICE_OP_H_

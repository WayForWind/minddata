

#include "OURSdata/dataset/text/kernels/data_utils.h"

#include <algorithm>
#include <string>

#include "OURSdata/dataset/core/pybind_support.h"
#include "OURSdata/dataset/kernels/data/slice_op.h"
#include "OURSdata/dataset/kernels/data/concatenate_op.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"

namespace ours {
namespace dataset {
Status SlidingWindowHelper(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, TensorShape out_shape,
                           uint32_t width, int32_t axis) {
  // if the data row has fewer items than width, the corresponding result row will be empty
  if (out_shape.Size() == 0) {
    MS_LOG(WARNING) << "The data row has fewer items than width, the result will be empty.";
    return Tensor::CreateEmpty(TensorShape({0}), input->type(), output);
  }

  axis = Tensor::HandleNeg(axis, input->shape().Size());
  int32_t axis_end = input->shape()[axis];
  std::shared_ptr<Tensor> tmp;
  auto concatenate_op = std::make_unique<ConcatenateOp>(axis, nullptr, nullptr);

  // Slice on specified axis and concatenate on new axis
  for (int32_t i = 0; i + width <= axis_end; i++) {
    auto slice_op = std::make_unique<SliceOp>(Slice(i, i + width, 1));
    RETURN_IF_NOT_OK(slice_op->Compute(input, &tmp));
    if (i == 0) {
      *output = tmp;
    } else {
      TensorRow in({*output, tmp});
      TensorRow out_row;
      RETURN_IF_NOT_OK(concatenate_op->Compute(in, &out_row));
      *output = out_row[0];
    }
  }
  RETURN_IF_NOT_OK((*output)->Reshape(out_shape));
  return Status::OK();
}

Status AppendOffsetsHelper(const std::vector<uint32_t> &offsets_start, const std::vector<uint32_t> &offsets_limit,
                           TensorRow *output) {
  std::shared_ptr<Tensor> offsets_start_tensor, offsets_limit_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(offsets_start, &offsets_start_tensor));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(offsets_limit, &offsets_limit_tensor));

  output->push_back(offsets_start_tensor);
  output->push_back(offsets_limit_tensor);
  return Status::OK();
}

Status AddToken(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const std::string &token,
                bool begin) {
  if (input->Rank() == 1) {
    std::shared_ptr<Tensor> append;
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(std::vector<std::string>({token}), &append));
    TensorRow in({input});
    TensorRow out;
    RETURN_IF_NOT_OK(Concatenate(in, &out, 0, begin ? append : nullptr, begin ? nullptr : append));
    *output = out[0];
  } else {
    std::vector<std::string> output_vector;
    int dim = input->shape()[0];
    int shape = input->shape()[-1];
    int count = 0;
    for (auto it = input->begin<std::string_view>(); it != input->end<std::string_view>(); ++it) {
      if (count >= shape) {
        count = 0;
      }
      if (begin && count == 0) {
        output_vector.emplace_back(token);
      }
      output_vector.emplace_back(*it);
      if (!begin && count == shape - 1) {
        output_vector.emplace_back(token);
      }
      count++;
    }
    shape++;
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(output_vector, TensorShape({dim, shape}), output));
  }
  return Status::OK();
}

Status Truncate(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int max_seq_len) {
  if (input->shape().Rank() == 1) {
    return input->Slice(output, {SliceOption(Slice(max_seq_len))});
  } else {
    int dim = input->shape()[0];
    Slice slice_dim = Slice(dim);
    Slice slice_len = Slice(max_seq_len);
    return input->Slice(output, {SliceOption(slice_dim), SliceOption(slice_len)});
  }
}
}  // namespace dataset
}  // namespace ours

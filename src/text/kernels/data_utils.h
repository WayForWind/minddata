
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_TEXT_DATA_UTILS_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_TEXT_DATA_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include "OURSdata/dataset/util/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/core/data_type.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/core/cv_tensor.h"
#include "OURSdata/dataset/core/tensor_shape.h"
#include "OURSdata/dataset/core/tensor_row.h"

namespace ours {
namespace dataset {
/// \brief Helper method that perform sliding window on input tensor.
/// \param[in] input - Input tensor.
/// \param[in] out_shape - Output shape of output tensor.
/// \param[in] width - The axis along which sliding window is computed.
/// \param[in] axis - The width of the window.
/// \param[out] output - Output tensor
/// \return Status return code
Status SlidingWindowHelper(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, TensorShape out_shape,
                           uint32_t width, int32_t axis);

/// \brief Helper method that append offsets tensor to output TensorRow.
/// \param[in] offsets_start - Offsets start index vector.
/// \param[in] offsets_limit - Offsets length vector.
/// \param[out] output - Output TensorRow
/// \return Status return code
Status AppendOffsetsHelper(const std::vector<uint32_t> &offsets_start, const std::vector<uint32_t> &offsets_limit,
                           TensorRow *output);

/// \brief Helper method that add token on input tensor.
/// \param[in] input Input tensor.
/// \param[in] token The token to be added.
/// \param[in] begin Whether to insert token at start or end of sequence.
/// \param[out] output Output tensor.
/// \return Status return code.
Status AddToken(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const std::string &token,
                bool begin);

/// \brief Truncate the input sequence so that it does not exceed the maximum length.
/// \param[in] max_seq_len Maximum allowable length.
/// \param[out] output Output Tensor.
Status Truncate(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int max_seq_len);
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_TEXT_DATA_UTILS_H_

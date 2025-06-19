

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_VALIDATORS_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_VALIDATORS_H_

#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"
#include "OURSdata/dataset/kernels/ir/validators.h"
#include "OURSdata/dataset/util/status.h"
#include "OURSdata/dataset/util/validators.h"

namespace ours {
namespace dataset {
// Helper function to positive int scalar
Status ValidateIntScalarNonNegative(const std::string &op_name, const std::string &scalar_name, int32_t scalar);

// Helper function to validate scalar value
template <typename T>
Status ValidateScalarValue(const std::string &op_name, const std::string &scalar_name, T scalar,
                           const std::vector<T> &values) {
  if (std::find(values.begin(), values.end(), scalar) == values.end()) {
    std::string init;
    std::string mode = std::accumulate(values.begin(), values.end(), init, [](const std::string &str, T val) {
      if (str.empty()) {
        return std::to_string(val);
      } else {
        return str + ", " + std::to_string(val);
      }
    });
    std::string err_msg =
      op_name + ": " + scalar_name + " must be one of [" + mode + "], but got: " + std::to_string(scalar);
    MS_LOG(ERROR) << err_msg;
    RETURN_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// Helper function to check scalar is not equal to zero
template <typename T>
Status ValidateScalarNotZero(const std::string &op_name, const std::string &scalar_name, const T scalar) {
  if (scalar == 0) {
    std::string err_msg =
      op_name + ": " + scalar_name + " can not be equal to zero, but got: " + std::to_string(scalar);
    MS_LOG(ERROR) << err_msg;
    RETURN_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// Helper function to check vector is not empty
template <typename T>
Status ValidateVectorNotEmpty(const std::string &op_name, const std::string &vec_name, const std::vector<T> &vec) {
  if (vec.empty()) {
    std::string err_msg = op_name + ": " + vec_name + " can not be an empty vector.";
    MS_LOG(ERROR) << err_msg;
    RETURN_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// Helper function to check two vector size equal
template <typename T>
Status ValidateVectorSameSize(const std::string &op_name, const std::string &vec_name, const std::vector<T> &vec,
                              const std::string &other_vec_name, const std::vector<T> &other_vec) {
  if (vec.size() != other_vec.size()) {
    std::string err_msg = op_name + ": the size of '" + vec_name + "' should be the same as that of '" +
                          other_vec_name + "', but got: '" + vec_name + "' size " + std::to_string(vec.size()) +
                          " and '" + other_vec_name + "' size " + std::to_string(other_vec.size()) + ".";
    MS_LOG(ERROR) << err_msg;
    RETURN_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_VALIDATORS_H_

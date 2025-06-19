
#include "OURSdata/dataset/audio/ir/validators.h"

namespace ours {
namespace dataset {
Status ValidateIntScalarNonNegative(const std::string &op_name, const std::string &scalar_name, int32_t scalar) {
  RETURN_IF_NOT_OK(ValidateScalar(op_name, scalar_name, scalar, {0}, false));
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours

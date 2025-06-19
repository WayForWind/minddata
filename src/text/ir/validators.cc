

#include "OURSdata/dataset/text/ir/validators.h"

namespace ours {
namespace dataset {
/* ####################################### Validator Functions ############################################ */

// Helper function to validate tokenizer directory parameter
Status ValidateTokenizerDirParam(const std::string &tokenizer_name, const std::string &tokenizer_file) {
  if (tokenizer_file.empty()) {
    std::string err_msg = tokenizer_name + ": tokenizer_file is not specified.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  Path file(tokenizer_file);
  if (!file.Exists()) {
    std::string err_msg = tokenizer_name + ": tokenizer_file: [" + tokenizer_file + "] is an invalid directory path.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (access(tokenizer_file.c_str(), R_OK) == -1) {
    std::string err_msg = tokenizer_name + ": No access to specified tokenizer path: " + tokenizer_file;
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

// Helper functions to help validate data type passed by user
bool IsTypeNumeric(const std::string &data_type) {
  if (data_type == "int8" || data_type == "uint8" || data_type == "int16" || data_type == "uint16" ||
      data_type == "int32" || data_type == "uint32" || data_type == "int64" || data_type == "uint64" ||
      data_type == "float16" || data_type == "float32" || data_type == "float64") {
    return true;
  }
  return false;
}

bool IsTypeBoolean(const std::string &data_type) { return data_type == "bool"; }

bool IsTypeString(const std::string &data_type) { return data_type == "string"; }
}  // namespace dataset
}  // namespace ours



#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_IR_VALIDATORS_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_IR_VALIDATORS_H_

#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
// Helper function to validate tokenizer directory parameter
Status ValidateTokenizerDirParam(const std::string &tokenizer_name, const std::string &tokenizer_file);

// Helper function to validate data type passed by user
bool IsTypeNumeric(const std::string &data_type);

// Helper function to validate data type is boolean
bool IsTypeBoolean(const std::string &data_type);

// Helper function to validate data type is string
bool IsTypeString(const std::string &data_type);
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_IR_VALIDATORS_H_

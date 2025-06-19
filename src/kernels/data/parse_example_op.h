/
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_PARSE_EXAMPLE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_PARSE_EXAMPLE_OP_H_

#include <unsupported/Eigen/CXX11/ThreadPool>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/engine/data_schema.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
constexpr int kThreadPoolSize = 32;

struct VarLenTensorBuffer {
  std::vector<std::shared_ptr<Tensor>> numeric_tensor;  // store the minibatch of numeric tensors
  std::vector<std::string> string_tensor;               // store the minibatch of strings
  size_t string_length;                                 // store the lengtn of string in minibatch
};

class ParseExampleOp : public TensorOp {
 public:
  ParseExampleOp(DataSchema data_schema, std::vector<std::string> column_list, bool parallel_parse)
      : data_schema_(std::move(data_schema)),
        column_list_(std::move(column_list)),
        parallel_parse_(parallel_parse),
        pool_(nullptr) {}

  ~ParseExampleOp() override = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kParseExampleOp; }

 private:
  Status ParseSingleExample(const TensorRow &raw_bytes, TensorRow *parsed_row);

  Status ParallelParseExample(const TensorRow &raw_bytes, TensorRow *parsed_row);

  Status ParseSerializedExample(const std::string &example_bytes, TensorRow *parsed_row,
                                std::unordered_map<int32_t, std::vector<std::string>> *string_column_map,
                                std::vector<VarLenTensorBuffer> *varlen_tensor_vector, size_t tensor_index);

  Status ConstructColumnMap(const std::string &example_bytes);

  void CheckAndInitPool();

  DataSchema data_schema_;
  std::vector<std::string> column_list_;
  bool parallel_parse_;
  std::unique_ptr<Eigen::ThreadPool> pool_;
  std::unordered_map<std::string, int32_t> column_name_id_map_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_PARSE_EXAMPLE_OP_H_

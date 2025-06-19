
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_OPERATOR_CONNECTOR_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_OPERATOR_CONNECTOR_H_

#include <memory>
#include <string>
#include <utility>
#include "OURSdata/dataset/core/tensor_row.h"
#include "OURSdata/dataset/engine/connector.h"

#include "OURSdata/dataset/include/dataset/constants.h"

namespace ours {
namespace dataset {

class OperatorConnector : public Queue<TensorRow> {
 public:
  /// Constructor of OperatorConnector
  /// \param queue_capacity The number of element (TensorRows) for the queue.
  explicit OperatorConnector(int32_t queue_capacity) : Queue<TensorRow>(queue_capacity), out_rows_count_(0) {}

  /// Destructor of -OperatorConnector
  ~OperatorConnector() = default;

  Status PopFront(TensorRow *row) override {
    out_rows_count_++;
    return Queue::PopFront(row);
  }

  Status SendEOE() noexcept {
    TensorRow eoe = TensorRow(TensorRow::kFlagEOE);
    return Add(std::move(eoe));
  }

  Status SendEOF() noexcept {
    TensorRow eof = TensorRow(TensorRow::kFlagEOF);
    return Add(std::move(eof));
  }

  Status SendEOB() noexcept {
    TensorRow eob = TensorRow(TensorRow::kFlagEOB);
    return Add(std::move(eob));
  }

  auto out_rows_count() const { return out_rows_count_; }

 private:
  int64_t out_rows_count_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_OPERATOR_CONNECTOR_H_

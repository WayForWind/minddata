
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_JAGGED_CONNECTOR_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_JAGGED_CONNECTOR_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "OURSdata/dataset/engine/connector.h"

#include "OURSdata/dataset/util/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"

namespace ours {
namespace dataset {
class JaggedConnector : public Connector<TensorRow> {
 public:
  JaggedConnector(int32_t num_producers, int32_t num_consumers, int32_t queue_capacity)
      : Connector<TensorRow>(num_producers, num_consumers, queue_capacity) {
    for (int i = 0; i < num_producers; i++) {
      is_queue_finished_.push_back(false);
    }
  }

  ~JaggedConnector() = default;

  Status Add(int32_t worker_d, TensorRow &&element) noexcept {
    return Connector<TensorRow>::Push(worker_d, std::move(element));
  }

  Status Pop(int32_t worker_id, TensorRow *result) noexcept override {
    RETURN_UNEXPECTED_IF_NULL(result);
    {
      MS_ASSERT(worker_id < num_consumers_);
      std::unique_lock<std::mutex> lock(m_);
      RETURN_IF_NOT_OK(cv_.Wait(&lock, [this, worker_id]() { return expect_consumer_ == worker_id; }));
      if (is_queue_finished_[pop_from_]) {
        std::string errMsg = "ERROR: popping from a finished queue in JaggedConnector";
        RETURN_STATUS_UNEXPECTED(errMsg);
      }

      RETURN_IF_NOT_OK(queues_[pop_from_]->PopFront(result));
      if (result != nullptr && result->eoe()) {
        is_queue_finished_[pop_from_] = true;
      }

      for (int offset = 1; offset <= num_producers_; offset++) {
        size_t nextQueueIndex = (pop_from_ + offset) % num_producers_;
        if (!is_queue_finished_[nextQueueIndex]) {
          pop_from_ = nextQueueIndex;
          break;
        }
      }

      expect_consumer_ = (expect_consumer_ + 1) % num_consumers_;
    }

    cv_.NotifyAll();
    return Status::OK();
  }

  void DoReset() {
    for (auto i = 0; i < is_queue_finished_.size(); i++) {
      is_queue_finished_[i] = false;
    }

    Connector<TensorRow>::Reset();
  }

 private:
  std::vector<bool> is_queue_finished_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_JAGGED_CONNECTOR_H_

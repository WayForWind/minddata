

#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_NGRAM_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_NGRAM_OP_H_

#include <string>
#include <memory>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class NgramOp : public TensorOp {
 public:
  // Constructor of Ngram model
  // @param const std::vector<int32_t> &ngrams
  // @param int32_t l_len - padding length on the left
  // @param const std::string &l_pad - padding token on the left
  // @param int32_t r_len - padding length on the right
  // @param const std::string &r_pad - padding token on the right
  // @param const std::string &separator - use to join strings
  NgramOp(const std::vector<int32_t> &ngrams, int32_t l_len, const std::string &l_pad, int32_t r_len,
          const std::string &r_pad, const std::string &separator);

  // perform ngram model on each tensor
  // @param const std::shared_ptr<Tensor> &input
  // @param std::shared_ptr<Tensor> *output
  // @return error code
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  // destructor
  ~NgramOp() override = default;

  // @param std::vector<TensorShape> &inputs - shape of input tensors
  // @param std::vector<TensorShape> &outputs - shape of output tensors
  // @return error code
  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  // print arg for debugging
  // @param std::ostream &out
  void Print(std::ostream &out) const override;

  std::string Name() const override { return kNgramOp; }

 private:
  std::vector<int32_t> ngrams_;  // list of n grams
  int32_t l_len_;                // left padding length
  int32_t r_len_;                // right padding length
  std::string l_pad_with_sp_;    // left padding appended with separator
  std::string r_pad_with_sp_;    // right padding appended with separator
  std::string separator_;        // separator
};

}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_NGRAM_OP_H_

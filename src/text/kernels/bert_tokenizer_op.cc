
#include "OURSdata/dataset/text/kernels/bert_tokenizer_op.h"
namespace ours {
namespace dataset {
Status BertTokenizerOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  TensorRow basic_tensor;
  RETURN_IF_NOT_OK(basic_tokenizer_.Compute(input, &basic_tensor));
  RETURN_IF_NOT_OK(wordpiece_tokenizer_.Compute(basic_tensor, output));
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours

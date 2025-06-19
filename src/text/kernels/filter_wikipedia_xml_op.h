/
#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_FILTER_WIKIPEDIA_XML_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_FILTER_WIKIPEDIA_XML_OP_H_

#include <map>
#include <memory>
#include <string>

#include "unicode/errorcode.h"
#include "unicode/regex.h"
#include "unicode/utypes.h"

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/text/kernels/whitespace_tokenizer_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class FilterWikipediaXMLOp : public TensorOp {
 public:
  FilterWikipediaXMLOp() {}

  ~FilterWikipediaXMLOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kFilterWikipediaXMLOp; }

 private:
  Status FilterWikipediaXML(const std::string_view &text, std::string *out) const;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_FILTER_WIKIPEDIA_XML_OP_H_

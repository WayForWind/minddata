
#include "OURSdata/dataset/engine/datasetops/source/fashion_mnist_op.h"

#include <fstream>

#include "OURSdata/dataset/core/config_manager.h"
#include "OURSdata/dataset/core/tensor_shape.h"
#include "OURSdata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "OURSdata/dataset/engine/execution_tree.h"
#include "utils/ms_utils.h"

namespace ours {
namespace dataset {
FashionMnistOp::FashionMnistOp(const std::string &usage, int32_t num_workers, const std::string &folder_path,
                               int32_t queue_size, std::unique_ptr<DataSchema> data_schema,
                               std::shared_ptr<SamplerRT> sampler)
    : MnistOp(usage, num_workers, folder_path, queue_size, std::move(data_schema), std::move(sampler)) {}

Status FashionMnistOp::CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count) {
  // the logic of counting the number of samples is copied from ParseMnistData() and uses CheckReader().
  RETURN_UNEXPECTED_IF_NULL(count);
  *count = 0;

  const int64_t num_samples = 0;
  const int64_t start_index = 0;
  auto sampler = std::make_shared<SequentialSamplerRT>(start_index, num_samples);
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  int32_t num_workers = cfg->num_parallel_workers();
  int32_t op_connect_size = cfg->op_connector_size();
  auto op =
    std::make_shared<FashionMnistOp>(usage, num_workers, dir, op_connect_size, std::move(schema), std::move(sampler));

  RETURN_IF_NOT_OK(op->WalkAllFiles());

  for (size_t i = 0; i < op->image_names_.size(); ++i) {
    std::ifstream image_reader;
    image_reader.open(op->image_names_[i], std::ios::in | std::ios::binary);
    CHECK_FAIL_RETURN_UNEXPECTED(image_reader.is_open(), "Invalid file, failed to open " + op->image_names_[i] +
                                                           ": the image file is damaged or permission denied.");
    std::ifstream label_reader;
    label_reader.open(op->label_names_[i], std::ios::in | std::ios::binary);
    CHECK_FAIL_RETURN_UNEXPECTED(label_reader.is_open(), "Invalid file, failed to open " + op->label_names_[i] +
                                                           ": the label file is damaged or permission denied.");
    uint32_t num_images;
    Status s = op->CheckImage(op->image_names_[i], &image_reader, &num_images);
    image_reader.close();
    RETURN_IF_NOT_OK(s);

    uint32_t num_labels;
    s = op->CheckLabel(op->label_names_[i], &label_reader, &num_labels);
    label_reader.close();
    RETURN_IF_NOT_OK(s);

    CHECK_FAIL_RETURN_UNEXPECTED(
      (num_images == num_labels),
      "Invalid data, num of images should be equal to num of labels, but got num of images: " +
        std::to_string(num_images) + ", num of labels: " + std::to_string(num_labels) + ".");
    *count = *count + num_images;
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace ours

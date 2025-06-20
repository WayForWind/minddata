

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_FLICKR_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_FLICKR_OP_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/engine/data_schema.h"
#include "OURSdata/dataset/engine/datasetops/parallel_op.h"
#include "OURSdata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "OURSdata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "OURSdata/dataset/util/path.h"
#include "OURSdata/dataset/util/queue.h"
#include "OURSdata/dataset/util/services.h"
#include "OURSdata/dataset/util/status.h"
#include "OURSdata/dataset/util/wait_post.h"

namespace ours {
namespace dataset {
class FlickrOp : public MappableLeafOp {
 public:
  /// \brief Constructor.
  /// \param[in] int32_t num_workers - Num of workers reading images in parallel
  /// \param[in] std::string dataset_dir - dir directory of Flickr dataset
  /// \param[in] std::string annotation_file - dir directory of annotation file
  /// \param[in] int32_t queue_size - connector queue size
  /// \param[in] std::unique_ptr<Sampler> sampler - sampler tells ImageFolderOp what to read
  FlickrOp(int32_t num_workers, const std::string &dataset_dir, const std::string &annotation_file, bool decode,
           int32_t queue_size, std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~FlickrOp() = default;

  /// \brief A print method typically used for debugging
  /// \param[out] out
  /// \param[in] show_all
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief Function to count the number of samples in the Flickr dataset
  /// \param[in] dir - path to the Flickr directory
  /// \param[in] file - path to the annotation file
  /// \param[out] count - output arg that will hold the actual dataset size
  /// \return Status - The status code returned
  static Status CountTotalRows(const std::string &dir, const std::string &file, int64_t *count);

  /// \brief Op name getter
  /// \return Name of the current Op
  std::string Name() const override { return "FlickrOp"; }

 protected:
  /// \brief Parse Flickr data
  /// \return Status - The status code returned
  Status PrepareData() override;

 private:
  /// \brief Load a tensor row according to a pair
  /// \param[in] uint64_t index - index need to load
  /// \param[out] TensorRow row - image & annotation read into this tensor row
  /// \return Status - The status code returned
  Status LoadTensorRow(row_id_type index, TensorRow *trow) override;

  /// \brief Check if image ia valid.Only support JPEG/PNG/GIF/BMP
  /// \param[in] std::string file_name - image file name need to be checked
  /// \param[out] bool valid - whether the image type is valid
  /// \return Status - The status code returned
  Status CheckImageType(const std::string &file_name, bool *valid);

  /// \brief Count annotation index,num rows and num samples
  /// \return Status - The status code returned
  Status CountDatasetInfo();

  /// \brief Private function for computing the assignment of the column name map.
  /// \return Status - The status code returned
  Status ComputeColMap() override;

  std::string dataset_dir_;
  std::string file_path_;
  bool decode_;
  std::unique_ptr<DataSchema> data_schema_;

  std::vector<std::pair<std::string, std::vector<std::string>>> image_annotation_pairs_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_FLICKR_OP_H_

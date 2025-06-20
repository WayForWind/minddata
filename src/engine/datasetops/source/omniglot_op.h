

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_OMNIGLOT_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_OMNIGLOT_OP_H_

#include <memory>
#include <queue>
#include <string>
#include <utility>

#include "OURSdata/dataset/engine/datasetops/source/image_folder_op.h"

namespace ours {
namespace dataset {
// Forward declares.
template <typename T>
class Queue;

using ImageLabelPair = std::shared_ptr<std::pair<std::string, int32_t>>;
using FolderImagesPair = std::shared_ptr<std::pair<std::string, std::queue<ImageLabelPair>>>;

class OmniglotOp : public ImageFolderOp {
 public:
  /// Constructor
  /// @param num_wkrs - Num of workers reading images in parallel.
  /// @param file_dir - Directory of ImageNetFolder.
  /// @param queue_size - Connector queue size.
  /// @param background - Use the background dataset or the evaluation dataset.
  /// @param do_decode - Decode the images after reading.
  /// @param data_schema - Schema of Omniglot dataset.
  /// @param sampler - Sampler tells OmniglotOp what to read.
  OmniglotOp(int32_t num_wkrs, const std::string &file_dir, int32_t queue_size, bool background, bool do_decode,
             std::unique_ptr<DataSchema> data_schema, const std::shared_ptr<SamplerRT> &sampler);

  /// Destructor.
  ~OmniglotOp() = default;

  /// A print method typically used for debugging.
  /// @param out - The output stream to write output to.
  /// @param show_all - A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  /// This is the common function to walk one directory.
  /// @param dir - The directory path
  /// @param folder_path - The queue in CountRowsAndClasses function.
  /// @param folder_name_queue - The queue in base class.
  /// @param dirname_offset - The offset of path of directory using in RecursiveWalkFolder function.
  /// @param std_queue - A bool to use folder_path or the foler_name_queue.
  /// @return Status - The error code returned.
  static Status WalkDir(Path *dir, std::queue<std::string> *folder_paths, Queue<std::string> *folder_name_queue,
                        uint64_t dirname_offset, bool std_queue);

  /// This function is a hack! It is to return the num_class and num_rows. The result
  /// returned by this function may not be consistent with what omniglot_op is going to return
  /// use this at your own risk!
  /// @param path - The folder path
  /// @param num_rows - The point to the number of rows
  /// @param num_classes - The point to the number of classes
  /// @return Status - the error code returned.
  static Status CountRowsAndClasses(const std::string &path, int64_t *num_rows, int64_t *num_classes);

  /// Op name getter
  /// @return std::string - Name of the current Op.
  std::string Name() const override { return "OmniglotOp"; }

  /// DatasetName name getter
  /// @param upper - A bool to control if you want to return uppercase or lowercase Op name.
  /// @return std::string - DatasetName of the current Op
  std::string DatasetName(bool upper = false) const { return upper ? "Omniglot" : "omniglot"; }

  /// Base-class override for GetNumClasses.
  /// @param num_classes - the number of classes.
  /// @return Status - the error code returned.
  Status GetNumClasses(int64_t *num_classes) override;

 private:
  //  Walk the folder
  /// @param dir - The folder path
  /// @return Status - the error code returned.
  Status RecursiveWalkFolder(Path *dir) override;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_OMNIGLOT_OP_H_

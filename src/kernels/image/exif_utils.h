

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_EXIF_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_EXIF_H_

namespace ours {
namespace dataset {

class ExifInfo {
 public:
  int parseOrientation(const unsigned char *data, unsigned len);
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_EXIF_H_

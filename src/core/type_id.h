
#ifndef OURS_CCSRC_OURSdata_DATASET_INCLUDE_TYPEID_H_
#define OURS_CCSRC_OURSdata_DATASET_INCLUDE_TYPEID_H_

#include "ir/dtype/type_id.h"
#include "OURSdata/dataset/core/data_type.h"

namespace ours {
namespace dataset {
inline dataset::DataType MSTypeToDEType(const TypeId data_type) {
  switch (data_type) {
    case kNumberTypeBool:
      return dataset::DataType(dataset::DataType::DE_BOOL);
    case kNumberTypeInt8:
      return dataset::DataType(dataset::DataType::DE_INT8);
    case kNumberTypeUInt8:
      return dataset::DataType(dataset::DataType::DE_UINT8);
    case kNumberTypeInt16:
      return dataset::DataType(dataset::DataType::DE_INT16);
    case kNumberTypeUInt16:
      return dataset::DataType(dataset::DataType::DE_UINT16);
    case kNumberTypeInt32:
      return dataset::DataType(dataset::DataType::DE_INT32);
    case kNumberTypeUInt32:
      return dataset::DataType(dataset::DataType::DE_UINT32);
    case kNumberTypeInt64:
      return dataset::DataType(dataset::DataType::DE_INT64);
    case kNumberTypeUInt64:
      return dataset::DataType(dataset::DataType::DE_UINT64);
    case kNumberTypeFloat16:
      return dataset::DataType(dataset::DataType::DE_FLOAT16);
    case kNumberTypeFloat32:
      return dataset::DataType(dataset::DataType::DE_FLOAT32);
    case kNumberTypeFloat64:
      return dataset::DataType(dataset::DataType::DE_FLOAT64);
    case kObjectTypeString:
      return dataset::DataType(dataset::DataType::DE_STRING);
    default:
      return dataset::DataType(dataset::DataType::DE_UNKNOWN);
  }
}

inline TypeId DETypeToMSType(dataset::DataType data_type) {
  switch (data_type.value()) {
    case dataset::DataType::DE_BOOL:
      return our::TypeId::kNumberTypeBool;
    case dataset::DataType::DE_INT8:
      return our::TypeId::kNumberTypeInt8;
    case dataset::DataType::DE_UINT8:
      return our::TypeId::kNumberTypeUInt8;
    case dataset::DataType::DE_INT16:
      return our::TypeId::kNumberTypeInt16;
    case dataset::DataType::DE_UINT16:
      return our::TypeId::kNumberTypeUInt16;
    case dataset::DataType::DE_INT32:
      return our::TypeId::kNumberTypeInt32;
    case dataset::DataType::DE_UINT32:
      return our::TypeId::kNumberTypeUInt32;
    case dataset::DataType::DE_INT64:
      return our::TypeId::kNumberTypeInt64;
    case dataset::DataType::DE_UINT64:
      return our::TypeId::kNumberTypeUInt64;
    case dataset::DataType::DE_FLOAT16:
      return our::TypeId::kNumberTypeFloat16;
    case dataset::DataType::DE_FLOAT32:
      return our::TypeId::kNumberTypeFloat32;
    case dataset::DataType::DE_FLOAT64:
      return our::TypeId::kNumberTypeFloat64;
    case dataset::DataType::DE_STRING:
      return our::TypeId::kObjectTypeString;
    default:
      return kTypeUnknown;
  }
}
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_INCLUDE_TYPEID_H_

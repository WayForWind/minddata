
#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "OURSdata/dataset/api/python/pybind_register.h"
#include "OURSdata/dataset/core/global_context.h"
#include "OURSdata/dataset/core/type_id.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/datasets.h"

namespace OURS {
namespace dataset {
PYBIND_REGISTER(
  SchemaObj, 0, ([](const py::module *m) {
    (void)py::class_<SchemaObj, std::shared_ptr<SchemaObj>>(*m, "SchemaObj", "to create a SchemaObj")
      .def(py::init([](const std::string &schema_file) {
        auto schema = std::make_shared<SchemaObj>(schema_file);
        THROW_IF_ERROR(schema->Init());
        return schema;
      }))
      .def("add_column",
           [](SchemaObj &self, const std::string &name, TypeId de_type, const std::vector<int32_t> &shape) {
             THROW_IF_ERROR(self.add_column(name, static_cast<OURS::DataType>(de_type), shape));
           })
      .def("add_column",
           [](SchemaObj &self, const std::string &name, const std::string &de_type, const std::vector<int32_t> &shape) {
             THROW_IF_ERROR(self.add_column(name, de_type, shape));
           })
      .def("add_column",
           [](SchemaObj &self, const std::string &name, TypeId de_type) {
             THROW_IF_ERROR(self.add_column(name, static_cast<OURS::DataType>(de_type)));
           })
      .def("add_column", [](SchemaObj &self, const std::string &name,
                            const std::string &de_type) { THROW_IF_ERROR(self.add_column(name, de_type)); })
      .def("parse_columns",
           [](SchemaObj &self, const std::string &json_string) { THROW_IF_ERROR(self.ParseColumnString(json_string)); })
      .def("to_json", &SchemaObj::to_json)
      .def("to_string", &SchemaObj::to_string)
      .def("from_string",
           [](SchemaObj &self, const std::string &json_string) { THROW_IF_ERROR(self.FromJSONString(json_string)); })
      .def("set_dataset_type",
           [](SchemaObj &self, const std::string &dataset_type) { self.set_dataset_type(dataset_type); })
      .def("set_num_rows", [](SchemaObj &self, int32_t num_rows) { self.set_num_rows(num_rows); })
      .def("get_num_rows", &SchemaObj::get_num_rows)
      .def("__deepcopy__", [](const py::object &schema, const py::dict &memo) { return schema; });
  }));
}  // namespace dataset
}  // namespace OURS


#include "pybind11/pybind11.h"

#include "OURSdata/dataset/api/python/pybind_conversion.h"
#include "OURSdata/dataset/api/python/pybind_register.h"
#include "OURSdata/dataset/core/type_id.h"
#include "OURSdata/dataset/include/dataset/execute.h"

namespace OURS {
namespace dataset {

PYBIND_REGISTER(Execute, 0, ([](const py::module *m) {
                  (void)py::class_<PyExecute, std::shared_ptr<PyExecute>>(*m, "Execute")
                    .def(py::init([](py::object operation) {
                      // current only support one op in python layer
                      auto execute = std::make_shared<PyExecute>(toTensorOperation(operation));
                      return execute;
                    }))
                    .def("UpdateOperation",
                         [](PyExecute &self, py::object operation) {
                           // update the op from python layer
                           THROW_IF_ERROR(self.UpdateOperation(toTensorOperation(operation)));
                         })
                    .def("__call__",
                         [](PyExecute &self, const std::vector<std::shared_ptr<Tensor>> &input_tensor_list) {
                           // Python API only supports cpu for eager mode
                           std::vector<std::shared_ptr<dataset::Tensor>> de_output_tensor_list;
                           THROW_IF_ERROR(self(input_tensor_list, &de_output_tensor_list));
                           return de_output_tensor_list;
                         });
                }));

}  // namespace dataset
}  // namespace OURS

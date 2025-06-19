
#include "pybind11/pybind11.h"

#include "OURSdata/dataset/api/python/pybind_register.h"
#include "OURSdata/dataset/core/global_context.h"

#include "OURS/ccsrc/OURSdata/dataset/kernels/data/compose_op.h"
#include "OURSdata/dataset/kernels/py_func_op.h"

namespace OURS {
namespace dataset {

Status PyListToTensorOps(const py::list &py_ops, std::vector<std::shared_ptr<TensorOp>> *ops) {
  RETURN_UNEXPECTED_IF_NULL(ops);
  for (auto op : py_ops) {
    if (py::isinstance<TensorOp>(op)) {
      ops->emplace_back(op.cast<std::shared_ptr<TensorOp>>());
    } else if (py::isinstance<py::function>(op)) {
      ops->emplace_back(std::make_shared<PyFuncOp>(op.cast<py::function>()));
    } else {
      RETURN_STATUS_UNEXPECTED("element is neither a TensorOp nor a pyfunc.");
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!ops->empty(), "TensorOp list is empty.");
  for (auto const &op : *ops) {
    RETURN_UNEXPECTED_IF_NULL(op);
  }
  return Status::OK();
}

PYBIND_REGISTER(TensorOp, 0, ([](const py::module *m) {
                  (void)py::class_<TensorOp, std::shared_ptr<TensorOp>>(*m, "TensorOp")
                    .def("__deepcopy__", [](py::object &t, py::dict memo) { return t; });
                }));
}  // namespace dataset
}  // namespace OURS

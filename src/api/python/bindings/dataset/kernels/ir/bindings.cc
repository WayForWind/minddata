
#include "pybind11/pybind11.h"

#include "OURS/ccsrc/OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/api/python/pybind_register.h"
#include "OURSdata/dataset/core/global_context.h"

#include "OURSdata/dataset/kernels/data/no_op.h"
#include "OURSdata/dataset/kernels/ir/data/transforms_ir.h"
#include "OURSdata/dataset/kernels/py_func_op.h"

namespace OURS {
namespace dataset {

Status PyListToTensorOperations(const py::list &py_ops, std::vector<std::shared_ptr<TensorOperation>> *ops) {
  RETURN_UNEXPECTED_IF_NULL(ops);
  for (auto op : py_ops) {
    if (py::isinstance<TensorOp>(op)) {
      ops->emplace_back(std::make_shared<transforms::PreBuiltOperation>(op.cast<std::shared_ptr<TensorOp>>()));
    } else if (py::isinstance<py::function>(op)) {
      ops->emplace_back(
        std::make_shared<transforms::PreBuiltOperation>(std::make_shared<PyFuncOp>(op.cast<py::function>())));
    } else if (py::isinstance<TensorOperation>(op)) {
      ops->emplace_back(op.cast<std::shared_ptr<TensorOperation>>());
    } else {
      RETURN_STATUS_UNEXPECTED("element is neither a TensorOp, TensorOperation nor a pyfunc.");
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!ops->empty(), "TensorOp list is empty.");
  for (auto const &op : *ops) {
    RETURN_UNEXPECTED_IF_NULL(op);
  }
  return Status::OK();
}

PYBIND_REGISTER(TensorOperation, 0, ([](const py::module *m) {
                  (void)py::class_<TensorOperation, std::shared_ptr<TensorOperation>>(*m, "TensorOperation");
                  py::arg("TensorOperation");
                }));

PYBIND_REGISTER(
  PluginOperation, 1, ([](const py::module *m) {
    (void)py::class_<transforms::PluginOperation, TensorOperation, std::shared_ptr<transforms::PluginOperation>>(
      *m, "PluginOperation")
      .def(py::init<std::string, std::string, std::string>());
  }));

PYBIND_REGISTER(NoOp, 1, ([](const py::module *m) {
                  (void)py::class_<NoOp, TensorOp, std::shared_ptr<NoOp>>(*m, "NoOp").def(py::init<>());
                }));

PYBIND_REGISTER(
  ComposeOperation, 1, ([](const py::module *m) {
    (void)py::class_<transforms::ComposeOperation, TensorOperation, std::shared_ptr<transforms::ComposeOperation>>(
      *m, "ComposeOperation")
      .def(py::init([](const py::list &ops) {
        std::vector<std::shared_ptr<TensorOperation>> t_ops;
        THROW_IF_ERROR(PyListToTensorOperations(ops, &t_ops));
        auto compose = std::make_shared<transforms::ComposeOperation>(std::move(t_ops));
        THROW_IF_ERROR(compose->ValidateParams());
        return compose;
      }));
  }));

PYBIND_REGISTER(
  ConcatenateOperation, 1, ([](const py::module *m) {
    (void)
      py::class_<transforms::ConcatenateOperation, TensorOperation, std::shared_ptr<transforms::ConcatenateOperation>>(
        *m, "ConcatenateOperation")
        .def(py::init([](int8_t axis, const std::shared_ptr<Tensor> &prepend, const std::shared_ptr<Tensor> &append) {
          auto concatenate = std::make_shared<transforms::ConcatenateOperation>(axis, prepend, append);
          THROW_IF_ERROR(concatenate->ValidateParams());
          return concatenate;
        }));
  }));

PYBIND_REGISTER(
  DuplicateOperation, 1, ([](const py::module *m) {
    (void)py::class_<transforms::DuplicateOperation, TensorOperation, std::shared_ptr<transforms::DuplicateOperation>>(
      *m, "DuplicateOperation")
      .def(py::init([]() {
        auto duplicate = std::make_shared<transforms::DuplicateOperation>();
        THROW_IF_ERROR(duplicate->ValidateParams());
        return duplicate;
      }));
  }));

PYBIND_REGISTER(FillOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<transforms::FillOperation, TensorOperation, std::shared_ptr<transforms::FillOperation>>(
                      *m, "FillOperation")
                      .def(py::init([](const std::shared_ptr<Tensor> &fill_value) {
                        auto fill = std::make_shared<transforms::FillOperation>(fill_value);
                        THROW_IF_ERROR(fill->ValidateParams());
                        return fill;
                      }));
                }));

PYBIND_REGISTER(MaskOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<transforms::MaskOperation, TensorOperation, std::shared_ptr<transforms::MaskOperation>>(
                      *m, "MaskOperation")
                      .def(py::init([](RelationalOp op, const std::shared_ptr<Tensor> &constant, DataType dtype) {
                        auto mask = std::make_shared<transforms::MaskOperation>(op, constant, dtype);
                        THROW_IF_ERROR(mask->ValidateParams());
                        return mask;
                      }));
                }));

PYBIND_REGISTER(
  OneHotOperation, 1, ([](const py::module *m) {
    (void)py::class_<transforms::OneHotOperation, TensorOperation, std::shared_ptr<transforms::OneHotOperation>>(
      *m, "OneHotOperation")
      .def(py::init([](int32_t num_classes, double smoothing_rate) {
        auto one_hot = std::make_shared<transforms::OneHotOperation>(num_classes, smoothing_rate);
        THROW_IF_ERROR(one_hot->ValidateParams());
        return one_hot;
      }));
  }));

PYBIND_REGISTER(
  PadEndOperation, 1, ([](const py::module *m) {
    (void)py::class_<transforms::PadEndOperation, TensorOperation, std::shared_ptr<transforms::PadEndOperation>>(
      *m, "PadEndOperation")
      .def(py::init([](TensorShape pad_shape, const std::shared_ptr<Tensor> &pad_value) {
        auto pad_end = std::make_shared<transforms::PadEndOperation>(pad_shape, pad_value);
        THROW_IF_ERROR(pad_end->ValidateParams());
        return pad_end;
      }));
  }));

PYBIND_REGISTER(RandomChoiceOperation, 1, ([](const py::module *m) {
                  (void)py::class_<transforms::RandomChoiceOperation, TensorOperation,
                                   std::shared_ptr<transforms::RandomChoiceOperation>>(*m, "RandomChoiceOperation")
                    .def(py::init([](const py::list &ops) {
                      std::vector<std::shared_ptr<TensorOperation>> t_ops;
                      THROW_IF_ERROR(PyListToTensorOperations(ops, &t_ops));
                      auto random_choice = std::make_shared<transforms::RandomChoiceOperation>(std::move(t_ops));
                      THROW_IF_ERROR(random_choice->ValidateParams());
                      return random_choice;
                    }));
                }));

PYBIND_REGISTER(RandomApplyOperation, 1, ([](const py::module *m) {
                  (void)py::class_<transforms::RandomApplyOperation, TensorOperation,
                                   std::shared_ptr<transforms::RandomApplyOperation>>(*m, "RandomApplyOperation")
                    .def(py::init([](double prob, const py::list &ops) {
                      std::vector<std::shared_ptr<TensorOperation>> t_ops;
                      THROW_IF_ERROR(PyListToTensorOperations(ops, &t_ops));
                      auto random_apply = std::make_shared<transforms::RandomApplyOperation>(std::move(t_ops), prob);
                      THROW_IF_ERROR(random_apply->ValidateParams());
                      return random_apply;
                    }));
                }));

PYBIND_REGISTER(
  SliceOperation, 1, ([](const py::module *m) {
    (void)py::class_<transforms::SliceOperation, TensorOperation, std::shared_ptr<transforms::SliceOperation>>(
      *m, "SliceOperation")
      .def(py::init([](std::vector<SliceOption> slice_input) {
        auto slice = std::make_shared<transforms::SliceOperation>(slice_input);
        THROW_IF_ERROR(slice->ValidateParams());
        return slice;
      }));
  }));

PYBIND_REGISTER(SliceOption, 0, ([](const py::module *m) {
                  (void)py::class_<SliceOption>(*m, "SliceOption")
                    .def(py::init([](const py::slice &py_slice) {
                      Slice c_slice;
                      if (!py_slice.attr("start").is_none() && !py_slice.attr("stop").is_none() &&
                          !py_slice.attr("step").is_none()) {
                        c_slice = Slice(py::reinterpret_borrow<py::int_>(py_slice.attr("start")),
                                        py::reinterpret_borrow<py::int_>(py_slice.attr("stop")),
                                        py::reinterpret_borrow<py::int_>(py_slice.attr("step")));
                      } else if (py_slice.attr("start").is_none() && py_slice.attr("step").is_none()) {
                        c_slice = Slice(py::reinterpret_borrow<py::int_>(py_slice.attr("stop")));
                      } else if (!py_slice.attr("start").is_none() && !py_slice.attr("stop").is_none()) {
                        c_slice = Slice(py::reinterpret_borrow<py::int_>(py_slice.attr("start")),
                                        py::reinterpret_borrow<py::int_>(py_slice.attr("stop")));
                      }

                      if (!c_slice.valid()) {
                        THROW_IF_ERROR(STATUS_ERROR(StatusCode::kMDUnexpectedError, "Wrong slice object"));
                      }
                      return SliceOption(c_slice);
                    }))
                    .def(py::init([](const py::list &py_list) {
                      std::vector<dsize_t> indices;
                      for (auto l : py_list) {
                        indices.push_back(py::reinterpret_borrow<py::int_>(l));
                      }
                      return SliceOption(indices);
                    }))
                    .def(py::init<bool>())
                    .def(py::init<SliceOption>());
                }));

PYBIND_REGISTER(
  TypeCastOperation, 1, ([](const py::module *m) {
    (void)py::class_<transforms::TypeCastOperation, TensorOperation, std::shared_ptr<transforms::TypeCastOperation>>(
      *m, "TypeCastOperation")
      .def(py::init([](const std::string &data_type) {
        auto type_cast = std::make_shared<transforms::TypeCastOperation>(data_type);
        THROW_IF_ERROR(type_cast->ValidateParams());
        return type_cast;
      }));
  }));

PYBIND_REGISTER(
  UniqueOperation, 1, ([](const py::module *m) {
    (void)py::class_<transforms::UniqueOperation, TensorOperation, std::shared_ptr<transforms::UniqueOperation>>(
      *m, "UniqueOperation")
      .def(py::init([]() {
        auto unique = std::make_shared<transforms::UniqueOperation>();
        THROW_IF_ERROR(unique->ValidateParams());
        return unique;
      }));
  }));

PYBIND_REGISTER(RelationalOp, 0, ([](const py::module *m) {
                  (void)py::enum_<RelationalOp>(*m, "RelationalOp", py::arithmetic())
                    .value("EQ", RelationalOp::kEqual)
                    .value("NE", RelationalOp::kNotEqual)
                    .value("LT", RelationalOp::kLess)
                    .value("LE", RelationalOp::kLessEqual)
                    .value("GT", RelationalOp::kGreater)
                    .value("GE", RelationalOp::kGreaterEqual)
                    .export_values();
                }));

}  // namespace dataset
}  // namespace OURS


#include "OURSdata/dataset/api/python/pybind_register.h"
#include "OURSdata/dataset/engine/datasetops/batch_op.h"

namespace OURS {
namespace dataset {

PYBIND_REGISTER(CBatchInfo, 0, ([](const py::module *m) {
                  (void)py::class_<CBatchInfo>(*m, "CBatchInfo")
                    .def(py::init<int64_t, int64_t, int64_t>())
                    .def("get_epoch_num", &CBatchInfo::get_epoch_num)
                    .def("get_batch_num", &CBatchInfo::get_batch_num)
                    .def(py::pickle(
                      [](const CBatchInfo &p) {  // __getstate__
                        /* Return a tuple that fully encodes the state of the object */
                        return py::make_tuple(p.epoch_num_, p.batch_num_, p.total_batch_num_);
                      },
                      [](py::tuple t) {  // __setstate__
                        if (t.size() != 3) throw std::runtime_error("Invalid state!");
                        /* Create a new C++ instance */
                        CBatchInfo p(t[0].cast<int64_t>(), t[1].cast<int64_t>(), t[2].cast<int64_t>());
                        return p;
                      }));
                }));

PYBIND_REGISTER(DatasetOp, 0, ([](const py::module *m) {
                  (void)py::class_<DatasetOp, std::shared_ptr<DatasetOp>>(*m, "DatasetOp");
                }));

}  // namespace dataset
}  // namespace OURS

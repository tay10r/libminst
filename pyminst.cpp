#include "minst.h"

#include <pybind11/pybind11.h>

#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

class sampler
{
public:
  virtual ~sampler() = default;

  virtual auto sample(uint32_t num_elements) -> uint32_t = 0;
};

class random_sampler final : public sampler
{
public:
  explicit random_sampler(const int seed)
    : m_rng(seed)
  {
  }

  auto sample(const uint32_t num_elements) -> uint32_t override
  {
    if (m_indices.size() != num_elements) {
      initialize_indices(num_elements);
      m_offset = 0;
    }

    if (m_offset == m_indices.size()) {
      shuffle_indices();
      m_offset = 0;
    }

    const auto idx = m_indices.at(m_offset);

    m_offset++;

    return idx;
  }

protected:
  using size_type = std::vector<uint32_t>::size_type;

  void initialize_indices(const uint32_t num_elements)
  {
    m_indices.resize(num_elements);

    for (uint32_t i = 0; i < num_elements; i++) {
      m_indices[i] = i;
    }

    shuffle_indices();
  }

  void shuffle_indices()
  {
    for (uint32_t i = 1; i < m_indices.size(); i++) {
      std::uniform_int_distribution<uint32_t> dist(0, i);
      const auto idx = dist(m_rng);
      const auto tmp = m_indices.at(i);
      m_indices.at(i) = m_indices.at(idx);
      m_indices.at(idx) = tmp;
    }
  }

private:
  std::mt19937 m_rng;

  std::vector<uint32_t> m_indices;

  size_type m_offset{};
};

class py_sampler : public sampler
{
public:
  auto sample(const uint32_t num_samples) -> uint32_t override
  {
    PYBIND11_OVERRIDE_PURE(uint32_t, sampler, sample, num_samples);
  }
};

int
call_sampler(void* sampler_ptr, const uint32_t num_samples, uint32_t* sample_idx)
{
  *sample_idx = static_cast<sampler*>(sampler_ptr)->sample(num_samples);
  return 0;
}

class callback
{
public:
  virtual ~callback() = default;

  virtual void eval(const py::bytes& sample, const py::bytes& label) = 0;
};

class py_callback : public callback
{
public:
  void eval(const py::bytes& sample, const py::bytes& label) override
  {
    PYBIND11_OVERRIDE_PURE(void, callback, eval, sample, label);
  }
};

struct format
{
  minst_type type{ MINST_TYPE_U8 };

  py::tuple shape;
};

struct callback_data final
{
  callback* cb{ nullptr };

  uint32_t sample_size{};

  uint32_t label_size{};
};

int
call(void* callback_ptr, const void* samples, const void* labels)
{
  auto* cb_data = static_cast<callback_data*>(callback_ptr);

  cb_data->cb->eval(py::bytes(static_cast<const char*>(samples), cb_data->sample_size),
                    py::bytes(static_cast<const char*>(labels), cb_data->label_size));

  return 0;
}

auto
to_c_format(const format& f) -> minst_format
{
  std::vector<uint32_t> shape;

  for (const auto& val : f.shape) {
    shape.emplace_back(val.cast<uint32_t>());
  }

  if (shape.size() > MINST_MAX_RANK) {
    std::ostringstream stream;
    stream << "Shape size of '" << shape.size() << "' exceeds maximum of '" << MINST_MAX_RANK << "'.";
    throw std::runtime_error(stream.str());
  }

  for (size_t i = shape.size(); i < MINST_MAX_RANK; i++) {
    shape.emplace_back(1);
  }

  minst_format fmt{};
  fmt.type = f.type;
  fmt.rank = f.shape.size();
  for (size_t i = 0; i < MINST_MAX_RANK; i++) {
    fmt.shape[i] = shape[i];
  }
  return fmt;
}

void
eval(const std::string& samples_path,
     const std::string& labels_path,
     const format& sample_format,
     const format& label_format,
     const uint32_t batch_size,
     callback& cb,
     sampler& s)
{
  const auto s_format = to_c_format(sample_format);
  const auto l_format = to_c_format(label_format);

  callback_data cb_data{ &cb, minst_element_size(&s_format) * batch_size, minst_element_size(&l_format) * batch_size };

  const auto err = minst_eval(
    samples_path.c_str(), labels_path.c_str(), &s_format, &l_format, batch_size, &cb_data, call, &s, call_sampler);

  if (err != MINST_ERR_NONE) {
    throw std::runtime_error(minst_strerror(err));
  }
}

} // namespace

PYBIND11_MODULE(pyminst, m)
{
  m.doc() = "A library for reading MINST datasets.";

  py::enum_<minst_type>(m, "Type")
    .value("U8", MINST_TYPE_U8, "An unsigned 8-bit integer.")
    .value("I8", MINST_TYPE_I8, "A signed 8-bit integer.")
    .value("I16", MINST_TYPE_I16, "A signed 16-bit integer.")
    .value("I32", MINST_TYPE_I32, "A signed 32-bit integer.")
    .value("F32", MINST_TYPE_F32, "A 32-bit floating point number.")
    .value("F64", MINST_TYPE_F64, "A 64-bit floating point number.");

  py::class_<format>(m, "Format")
    .def(py::init<>())
    .def_readwrite("shape", &format::shape, "The shape of the tensor.")
    .def_readwrite("type_", &format::type, "The type of the tensor coefficients.");

  py::class_<sampler, py_sampler>(m, "Sampler")
    .def(py::init<>())
    .def("sample", &sampler::sample, py::arg("num_samples"));

  py::class_<random_sampler, sampler>(m, "RandomSampler").def(py::init<int>());

  py::class_<callback, py_callback>(m, "Callback")
    .def(py::init<>())
    .def("eval", &callback::eval, py::arg("sample"), py::arg("label"));

  m.def("eval",
        eval,
        "Iterates a dataset.",
        py::arg("samples_path"),
        py::arg("labels_path"),
        py::arg("sample_format"),
        py::arg("label_format"),
        py::arg("batch_size"),
        py::arg("callback"),
        py::arg("sampler"));
}

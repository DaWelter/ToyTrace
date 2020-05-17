#include "vec3f.hxx"
#include "util.hxx"
#include "span.hxx"
#include "sampler.hxx"
#include "distribution_mixture_models.hxx"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

namespace 
{

template<class T>
py::object CastToNumpyArray(const T &a)
{
    // Pybind will want to generate lists for std::array<T,N>.
    // This is fixed here. "asarray" will collapse nested lists into
    // a multidimensional array. Ofc, the input needs to have a 
    // apropriate structure.
    return py::module::import("numpy").attr("asarray")(py::cast(a));
}

}

// Careful with alignment and auto-vectorization. (Should be okay because Eigen does not auto-vectorize Vector2f. Or does it?)
using DataPointArray = std::vector<Eigen::Vector2f>;
using DataPointArray3d = std::vector<Eigen::Vector3f>;
using DataWeightArray = std::vector<float>;

namespace py_vmf_mixture
{

using namespace vmf_fitting;

template<int N>
struct PyVonMisesFischerMixture : public VonMisesFischerMixture<N>
{
    auto pdf(DataPointArray3d xs)
    {
        DataWeightArray result; result.reserve(xs.size());
        for (const auto &x : xs)
        {
            result.push_back(vmf_fitting::Pdf(*this, x));
        }
        return CastToNumpyArray(result);
    }

    auto sample(int n)
    {
        DataPointArray3d result; result.reserve(n);

        auto np = py::module::import("numpy");
        auto np_random = np.attr("random").attr("random");
        auto py_random_vals = np_random(py::make_tuple(n, 3));
        auto rs = py_random_vals.cast<py::array_t<double>>().unchecked<2>();

        for (int i = 0; i < n; ++i)
        {
            const auto x = vmf_fitting::Sample(*this, { rs(i,0), rs(i,1), rs(i,2) });
            result.push_back(x);
        }

        return CastToNumpyArray(result);
    }

    void setWeights(const typename VonMisesFischerMixture<N>::WeightArray &a)
    {
        this->weights = a;
    }

    auto getWeights()
    {
        return CastToNumpyArray(this->weights);
    }

    void setMeans(const typename VonMisesFischerMixture<N>::MeansArray &a)
    {
        this->means = a;
    }

    auto getMeans()
    {
        return CastToNumpyArray(this->means);
    }

    void setConcentrations(const typename VonMisesFischerMixture<N>::ConcArray &a)
    {
        this->concentrations = a;
    }

    auto getConcentrations()
    {
        return CastToNumpyArray(this->concentrations);
    }

    static void Register(py::module &m)
    {
        py::class_<PyVonMisesFischerMixture<N>>(m, strconcat("VMFMixture",N).c_str())
            .def(py::init<>([]() -> PyVonMisesFischerMixture<N> { 
                PyVonMisesFischerMixture<N> ret; 
                InitializeForUnitSphere(ret); 
                return std::move(ret);
            }))
            .def("pdf", &PyVonMisesFischerMixture<N>::pdf)
            .def("sample", &PyVonMisesFischerMixture<N>::sample)
            .def_property("weights", &PyVonMisesFischerMixture<N>::getWeights, &PyVonMisesFischerMixture<N>::setWeights)
            .def_property("means", &PyVonMisesFischerMixture<N>::getMeans, &PyVonMisesFischerMixture<N>::setMeans)
            .def_property("concentrations", &PyVonMisesFischerMixture<N>::getConcentrations, &PyVonMisesFischerMixture<N>::setConcentrations);
    }
};



template<int N>
class PyMoVmfFitIncremental
{
    incremental::Data<N> data;
    incremental::Params<N> params;
    PyVonMisesFischerMixture<N> prior_mode;

    static vmf_fitting::incremental::Params<N> makeParams(py::dict d)
    {
        incremental::Params<N> p;
        p.prior_nu = d["prior_nu"].cast<float>();
        p.prior_alpha = d["prior_alpha"].cast<float>();
        p.prior_tau = d["prior_tau"].cast<float>();
        p.maximization_step_every = d["maximization_step_every"].cast<int>();
        return p;
    }

public:
    PyMoVmfFitIncremental(py::kwargs kwargs) :
        params{makeParams(kwargs)},
        prior_mode{kwargs["prior_mode"].cast<PyVonMisesFischerMixture<N>>()}
    {
      params.prior_mode = &prior_mode;
    }

    void Fit(VonMisesFischerMixture<N> &mixture, 
        const DataPointArray3d xs, 
        const DataWeightArray ws)
    {            
        incremental::Fit(mixture, data, params, AsSpan(xs), AsSpan(ws));
    }

    static void Register(py::module &m)
    {
        py::class_<PyMoVmfFitIncremental<N>>(m, strconcat("VMFFitIncremental", N).c_str())
        .def(py::init<py::kwargs>())
        .def("fit", &PyMoVmfFitIncremental::Fit);
    }
};



}



PYBIND11_MODULE(path_guiding, m)
{
    py::module::import("numpy");

    m.doc() = "bindings for debugging path guiding";

    py_vmf_mixture::PyVonMisesFischerMixture<1>::Register(m);
    py_vmf_mixture::PyVonMisesFischerMixture<2>::Register(m);
    py_vmf_mixture::PyVonMisesFischerMixture<8>::Register(m);
    py_vmf_mixture::PyMoVmfFitIncremental<1>::Register(m);
    py_vmf_mixture::PyMoVmfFitIncremental<2>::Register(m);
    py_vmf_mixture::PyMoVmfFitIncremental<8>::Register(m);
}
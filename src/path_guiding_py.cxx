#include "vec3f.hxx"
#include "util.hxx"
#include "span.hxx"
#include "sampler.hxx"
#include "distribution_mixture_models.hxx"
#include "shader.hxx"
#include "path_guiding_quadtree.hxx"
#include "shader_util.hxx"

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

namespace guiding::quadtree_radiance_distribution
{

Eigen::Vector2f MapSphereToTree(const Eigen::Vector3f &dir);
Eigen::Vector3f MapTreeToSphere(const Eigen::Vector2f &uv);

}


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

template<class T>
auto CopyToStdVector(const ToyVector<T> &x)
{
    return std::vector<T>{ x.begin(), x.end() };
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
auto pdf(VonMisesFischerMixture<N> &self, DataPointArray3d xs)
{
    DataWeightArray result; result.reserve(xs.size());
    for (const auto &x : xs)
    {
        result.push_back(vmf_fitting::Pdf(self, x));
    }
    return CastToNumpyArray(result);
}

template<int N>
auto sample(VonMisesFischerMixture<N> &self, int n)
{
    DataPointArray3d result; result.reserve(n);

    auto np = py::module::import("numpy");
    auto np_random = np.attr("random").attr("random");
    auto py_random_vals = np_random(py::make_tuple(n, 3));
    auto rs = py_random_vals.cast<py::array_t<double>>().unchecked<2>();

    for (int i = 0; i < n; ++i)
    {
        const auto x = vmf_fitting::Sample(self, { rs(i,0), rs(i,1), rs(i,2) });
        result.push_back(x);
    }

    return CastToNumpyArray(result);
}

template<int N>
void setWeights(VonMisesFischerMixture<N> &self, const typename VonMisesFischerMixture<N>::WeightArray &a)
{
    self.weights = a;
}

template<int N>
auto getWeights(VonMisesFischerMixture<N> &self)
{
    return CastToNumpyArray(self.weights);
}

template<int N>
void setMeans(VonMisesFischerMixture<N> &self, const typename VonMisesFischerMixture<N>::MeansArray &a)
{
    self.means = a;
}

template<int N>
auto getMeans(VonMisesFischerMixture<N> &self)
{
    return CastToNumpyArray(self.means);
}

template<int N>
void setConcentrations(VonMisesFischerMixture<N> &self, const typename VonMisesFischerMixture<N>::ConcArray &a)
{
    self.concentrations = a;
}

template<int N>
auto getConcentrations(VonMisesFischerMixture<N> &self)
{
    return CastToNumpyArray(self.concentrations);
}

template<int N>
void Register(py::module &m)
{
    py::class_<VonMisesFischerMixture<N>>(m, fmt::format("VMFMixture{}",N).c_str())
        .def(py::init<>([]() -> VonMisesFischerMixture<N> { 
            VonMisesFischerMixture<N> ret; 
            InitializeForUnitSphere(ret); 
            return std::move(ret);
        }))
        .def("pdf", &pdf<N>)
        .def("sample", &sample<N>)
        .def_property("weights", &getWeights<N>, &setWeights<N>)
        .def_property("means", &getMeans<N>, &setMeans<N>)
        .def_property("concentrations", &getConcentrations<N>, &setConcentrations<N>);
}


template<int N, int M>
void RegisterBinaryOps(py::module &m)
{
    m.def("product", &vmf_fitting::Product<N, M>);
}




template<int N>
class PyMoVmfFitIncremental
{
    incremental::Data<N> data;
    incremental::Params<N> params;
    VonMisesFischerMixture<N> prior_mode;

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
        prior_mode{kwargs["prior_mode"].cast<VonMisesFischerMixture<N>>()}
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
        py::class_<PyMoVmfFitIncremental<N>>(m, fmt::format("VMFFitIncremental{}", N).c_str())
        .def(py::init<py::kwargs>())
        .def("fit", &PyMoVmfFitIncremental::Fit);
    }
};


}


namespace py_shady
{


void RegisterSurfaceInteraction(py::module &m)
{
    py::class_<SurfaceInteraction>(m, "SurfaceInteraction")
    .def(py::init<>([](
        const Eigen::Vector3d &geometry_normal,
        const Eigen::Vector3d &smooth_normal) -> SurfaceInteraction {
        SurfaceInteraction si{};
        si.geometry_normal = geometry_normal;
        si.smooth_normal = smooth_normal,
        // Set the incident-aligned normals, assuming the incident ray comes 
        // from the hemisphere that the geo-normal points to.
        si.SetOrientedNormals(-geometry_normal);
        return si;
    }));
}


struct PyShaderWrapper
{
    PyShaderWrapper(std::unique_ptr<Shader>&& shader)
        : shader{std::move(shader)}, context{SelectRgbPrimaryWavelengths()}
    {
    }

    std::unique_ptr<Shader> shader;
    Sampler sampler; // Each to his own. So I don't have to wrap this, too.
    PathContext context;
};


std::tuple<Double3, Spectral3, double, bool> Sample(PyShaderWrapper &pyshady, const Eigen::Vector3d &incident_dir, const SurfaceInteraction &surface_hit)
{
    auto smpl = pyshady.shader->SampleBSDF(incident_dir, surface_hit, pyshady.sampler, pyshady.context);
    return std::make_tuple(smpl.coordinates, smpl.value, (double)smpl.pdf_or_pmf, smpl.pdf_or_pmf.IsFromDelta());
}

Spectral3 Evaluate(PyShaderWrapper &pyshady, const Eigen::Vector3d &incident_dir, const SurfaceInteraction &surface_hit, const Eigen::Vector3d &out_direction)
{
    return pyshady.shader->EvaluateBSDF(incident_dir, surface_hit, out_direction, pyshady.context, nullptr);
}

#ifdef PRODUCT_DISTRIBUTION_SAMPLING
py_vmf_mixture::VonMisesFischerMixture<2> ComputeLobes(PyShaderWrapper &pyshady, const Eigen::Vector3d &incident_dir, const SurfaceInteraction &surface_hit)
{
    return pyshady.shader->ComputeLobes(incident_dir, surface_hit, pyshady.context);
}
#endif

// double EvaluatePdf(PyShaderWrapper &pyshady, const Double3 &incident_dir, const SurfaceInteraction &surface_hit, const Double3 &out_direction)
// {

// }


void RegisterShaders(py::module &m)
{
  py::class_<PyShaderWrapper>(m, "PyShaderWrapper")
    .def("Sample", Sample)
    .def("Evaluate", Evaluate)
#ifdef PRODUCT_DISTRIBUTION_SAMPLING
    .def("InitializeLobes", [](PyShaderWrapper &pyshady) { pyshady.shader->IntializeLobes(); })
    .def("ComputeLobes", ComputeLobes)
#endif
    ;

    m.def("DiffuseShader", [](const SpectralN &reflectance) {
      return PyShaderWrapper{ MakeDiffuseShader(reflectance, nullptr) };
    });
    m.def("GlossyTransmissiveDielectricShader", [](double ior_ratio,  double alpha) {
        //return PyShaderWrapper::make<GlossyTransmissiveDielectricShader>(ior_ratio, alpha, 0., nullptr);
        return PyShaderWrapper{MakeGlossyTransmissiveDielectricShader(ior_ratio, alpha, 0., nullptr)};
    });
}


} // namespace py_shady



namespace py_quadtree
{

using namespace guiding::quadtree;


void Register(py::module &m)
{
  py::class_<Tree>(m, "QuadTree")
  .def(py::init<>())
  .def(py::init<>([](const Eigen::ArrayX4i &a, int b) {
      return Tree(a, b);
  }))
  .def("Adapted", [](const Tree &tree, std::vector<float> weights, float weight_fraction_threshold) -> py::tuple {
    TreeAdaptor adaptor{ tree, AsSpan(weights), weight_fraction_threshold };
    auto new_weights = CopyToStdVector(adaptor.ExtractWeights());
    return py::make_tuple(adaptor.ExtractTree(), CastToNumpyArray(new_weights));
  })
  .def_static("Build", [](std::vector<Eigen::Vector2f> points, std::vector<float> weights, float weight_fraction_threshold) -> py::tuple {
    Builder builder{ AsSpan(points), AsSpan(weights), weight_fraction_threshold };
    auto new_weights = CopyToStdVector(builder.ExtractWeights());
    return py::make_tuple(builder.ExtractTree(), CastToNumpyArray(new_weights));
  })
  .def("PushWeights", [](const Tree &tree, py::array_t<float> py_node_weights, std::vector<Eigen::Vector2f> points, std::vector<float> weights) {
   auto a = py_node_weights.mutable_unchecked<1>();
   Span<float> node_weights{ a.mutable_data(0), a.size() };
   for (long i = 0; i < lsize(points); ++i)
   {
     PushWeight(tree, node_weights, points[i], weights[i]);
   }
  })
  .def("GenerateQuads", [](const Tree &tree) {
   return CastToNumpyArray(CopyToStdVector(GenerateQuads(tree)));
  })
  .def("GenerateLevels", [](const Tree &tree) {
   return CastToNumpyArray(CopyToStdVector(GenerateLevels(tree)));
  })
  .def("NumNodes", &Tree::NumNodes)
  .def("IsLeaf", [](const Tree &tree, int node) { 
      return tree.IsLeaf(node);
  })
  .def("Sample", [](const Tree &tree, std::vector<float> weights, int n) {
    Eigen::Array<float, Eigen::Dynamic, 2> points;
    Eigen::Array<float, Eigen::Dynamic, 1> pdfs;
    points.resize(n, 2);
    pdfs.resize(n, 1);
    Sampler sampler;
    for (int i=0; i<n; ++i)
    {
        auto [p, pdf] = guiding::quadtree::detail::Sample(tree, AsSpan(weights), sampler);
        points.row(i) = p.transpose();
        pdfs.row(i) = pdf;
    }
    return py::make_tuple(points, pdfs);
  })
//   .def("Sample", [](const Tree &tree, std::vector<float> weights, Eigen::ArrayX2f rnds) {
//     Eigen::Array<float, Eigen::Dynamic, 2> points;
//     Eigen::Array<float, Eigen::Dynamic, 1> pdfs;
//     points.resize(rnds.rows(), 2);
//     pdfs.resize(rnds.rows(), 1);
//     for (int i=0; i<rnds.rows(); ++i)
//     {
//         auto [p, pdf] = guiding::quadtree::detail::Sample(tree, AsSpan(weights), rnds.row(i));
//         points.row(i) = p.transpose();
//         pdfs.row(i) = pdf;
//     }
//     return py::make_tuple(points, pdfs);
//   })
  .def("Pdf", [](const Tree &tree, std::vector<Eigen::Vector2f> points, std::vector<float> weights) {
      std::vector<float> pdfs; pdfs.reserve(points.size());
      for (const auto &pt : points)
        pdfs.push_back(guiding::quadtree::detail::Pdf(tree, AsSpan(weights), pt));
    return CastToNumpyArray(pdfs);
  });

  m.def("MapSphereToTree", [](const Eigen::ArrayX3f &xs) {
    Eigen::ArrayX2f result(xs.rows(), 2);
    for (int i=0; i<xs.rows(); ++i)
        result.row(i) = guiding::quadtree_radiance_distribution::MapSphereToTree(xs.row(i).matrix()).array();
    return result;
  });
  //m.def("MapTreeToSphere", &guiding::quadtree_radiance_distribution::MapTreeToSphere);
}

} // py_quadtree


PYBIND11_MODULE(path_guiding, m)
{
    py::module::import("numpy");

    m.doc() = "bindings for debugging path guiding";

    py_vmf_mixture::Register<2>(m);
    py_vmf_mixture::Register<8>(m);
    py_vmf_mixture::Register<16>(m);
    py_vmf_mixture::PyMoVmfFitIncremental<2>::Register(m);
    py_vmf_mixture::PyMoVmfFitIncremental<8>::Register(m);
    py_vmf_mixture::RegisterBinaryOps<2,8>(m);

    py_shady::RegisterShaders(m);
    py_shady::RegisterSurfaceInteraction(m);
    
    m.def("ExpApproximation", [](const Eigen::Array<float, 8, 1> &vals) -> Eigen::Array<float, 8, 1>
    {
        Eigen::Array<float, 8, 1> result = vals;
        vmf_fitting::ExpApproximation<8>(result);
        return result;
    });
    m.def("ExpApproximation", [](float x) { return vmf_fitting::ExpApproximation(x);  });

    py_quadtree::Register(m);
}
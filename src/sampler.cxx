#include "sampler.hxx"

namespace SampleTrafo
{
// Ref: Global Illumination Compendium (2003)
Double3 ToUniformDisc(Double2 r)
{
  double s = std::sqrt(r[1]);
  double omega = 2.*Pi*r[0];
  double x = s*std::cos(omega);
  double y = s*std::sin(omega);
  return Double3{x,y,0};
}

// Ref: Global Illumination Compendium (2003)
Double3 ToUniformSphere(Double2 r)
{
  double z = 1. - 2.*r[1];
  double s = std::sqrt(r[1]*(1.-r[1]));
  double omega = 2.*Pi*r[0];
  double sn = std::sin(omega);
  double cs = std::cos(omega);
  double x = 2.*cs*s;
  double y = 2.*sn*s;
  return Double3{x,y,z};
}


// Modification of the above
Double3 ToUniformSphereSection(Double2 r, double phi0, double z0, double phi1, double z1)
{
//   const double z0 = std::cos(theta0);
//   const double z1 = std::cos(theta1);
  const double z = z0 + r[1]*(z1 - z0);
  double s = std::sqrt(1. - z*z);
  s = std::isnan(s) ? 0. : s;
  double omega = phi0 + (phi1-phi0)*r[0];
  double sn = std::sin(omega);
  double cs = std::cos(omega);
  double x = cs*s;
  double y = sn*s;
  return Double3{x,y,z};
}


// That one is obvious isn't it.
Double3 ToUniformHemisphere(Double2 r)
{
  Double3 v = ToUniformSphere(r);
  v[2] = v[2]>=0. ? v[2] : -v[2];
  return v;
}

// Ref: Global Illumination Compendium (2003)
Double3 ToUniformSphereSection(double cos_opening_angle, Double2 r)
{
  double z = 1.-r[1]*(1.-cos_opening_angle);
  double s = std::sqrt(1. - z*z);
  double omega = 2.*Pi*r[0];
  double sn = std::sin(omega);
  double cs = std::cos(omega);
  double x = cs*s;
  double y = sn*s;
  return Double3{x,y,z};
}

double UniformSphereSectionPdf(double cos_opening_angle)
{
  return 1. / (2.*Pi*(1. - cos_opening_angle));
}


// Ref: Global Illumination Compendium (2003)
Double3 ToCosHemisphere(Double2 r)
{
  double rho = std::sqrt(1.-r[0]);
  double z   = std::sqrt(r[0]);
  double omega = 2.*Pi*r[1];
  double sn = std::sin(omega);
  double cs = std::cos(omega);
  return Double3{cs*rho, sn*rho, z};
}

Double3 ToPhongHemisphere(Double2 r, double alpha)
{
  double t = std::pow(r[0], 1./(alpha+1));
  double rho = std::sqrt(1.-t);
  double z   = std::sqrt(t);
  double omega = 2.*Pi*r[1];
  double sn = std::sin(omega);
  double cs = std::cos(omega);
  return Double3{cs*rho, sn*rho, z};
}


/* Ref: Total Compendium pg. 12 */
Double3 ToTriangleBarycentricCoords(Double2 r)
{
  double sqrtr0 = std::sqrt(r[0]);
  double alpha = 1. - sqrtr0;
  double beta  = (1.-r[1])*sqrtr0;
  double gamma = r[1]*sqrtr0;
  return Double3{alpha, beta, gamma};
}


Double3 ToUniformSphere3d(const Double3 &rvs)
{
// https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
// Apply inversion method to probability of finding a point at distance D in a smaller sphere of radius 'rho'.
// That is, CDF(rho) = P(D < rho) = (rho/sphere_radius)^3.
// Thus, rho = CDF^{-1}(rvs[0]) = ....
  double rho = std::pow(rvs[0], 1./3.);
  Double3 p = ToUniformSphere({rvs[1], rvs[2]});
  return rho*p;
}


Float2 ToNormal2d(const Float2 &r)
{
    auto n1 = std::sqrt(-2 * std::log(r[0])) * std::cos(2 * float(Pi) * r[1]);
    auto n2 = std::sqrt(-2 * std::log(r[0])) * std::sin(2 * float(Pi) * r[1]);
    return { n1, n2 };
}


}


namespace
{
// Copy pasted from Kollig & Keller 2012 "Efficient Multidimensional Sampling"

using uint = std::uint32_t;

uint ReverseBits(uint bits)
{
  bits = (bits << 16)
    | (bits >> 16);
  bits = ((bits & 0x00ff00ff) << 8)
    | ((bits & 0xff00ff00) >> 8);
  bits = ((bits & 0x0f0f0f0f) << 4)
    | ((bits & 0xf0f0f0f0) >> 4);
  bits = ((bits & 0x33333333) << 2)
    | ((bits & 0xcccccccc) >> 2);
  bits = ((bits & 0x55555555) << 1)
    | ((bits & 0xaaaaaaaa) >> 1);
  return bits;
}

double RI_vdC(uint bits, uint r = 0)
{
  bits = ReverseBits(bits);
  bits ^= r;
  return (double)bits / (double)0x100000000LL;
}

double RI_S(uint i, uint r = 0)
{
  for (uint v = 1 << 31; i; i >>= 1, v ^= v >> 1)
    if (i & 1)
      r ^= v;
  return (double)r / (double)0x100000000LL;
}


//From here https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
unsigned int hash(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}
/*
// From "Stratified Sampling for Stochastic Transparency"
2: x ← reverseBits(x) + surfaceId
3: x ← x ⊕ (x ⋅ 0x6C50B47Cu )
4: x ← x ⊕ (x ⋅ 0xB82F1E52u )
5: x ← x ⊕ (x ⋅ 0xC7AFE638u )
6: x ← x ⊕ (x ⋅ 0x8D22F6E6u )
7: return x scaled to range [0, 1]
*/
uint hash2(uint bits)
{
   bits ^= bits * 0x6C50B47Cu;
   bits ^= bits * 0xB82F1E52u;
   bits ^= bits * 0xC7AFE638u;
   bits ^= bits * 0x8D22F6E6u;
   return bits;
}

uint hash3(uint bits)
{
  // Just other constants
   bits ^= bits * 0xd169f4d6u;
   bits ^= bits * 0x2cbb59c8u;
   bits ^= bits * 0xb64b806cu;
   bits ^= bits * 0xa0836df0u;
   return bits;
}

double AsUniform01(uint x)
{
  return (double)x / (double)0x100000000LL;
}

}


static constexpr int rotation_block_size = 128;
extern Double2 rotation_offsets[rotation_block_size*rotation_block_size];

class QuasiRandomSequence : public SampleSequence
{
  static constexpr int rotation_block_size = 64;
  static Double2 rotation_offsets[rotation_block_size*rotation_block_size];
  //using OffsetTable2d = ToyVector<Double2>;
  //OffsetTable2d rotation_offsets;
  int rotation_idx = 0;
  int point_idx = 0;
  uint32_t scrambling_mask = 0;
  Eigen::Vector2d subsequence_offsets{Eigen::zero};
  std::unordered_map<uint32_t, uint32_t> scrambling_masks;
  RandGen rnd;
public:
  QuasiRandomSequence()
  {
  }

  void SetPixelIndex(Int2 pixel_coord) override
  {
    rotation_idx = 
      (pixel_coord[0] % rotation_block_size) +
      (pixel_coord[1] % rotation_block_size) * rotation_block_size;
  }
  
  void SetPointNum(int i) override
  {
    assert(i >= 0);
    point_idx = i;
  }
  

  void SetSubsequenceId(uint32_t id) override
  {
    uint32_t m = scrambling_masks[id];
    if (m == 0)
    {
      // Scrambling masks need to be pretty random.
      // Otherwise there will be correlations across dimensions,
      // leading to noticable errors or image artifacts.
      m = rnd.GetGenerator().nextUInt();
      scrambling_masks[id] = m;
    }
    scrambling_mask = m;
    // Hashing is not too bad either. Not as good as random numbers though.
    // Can use rotation in addition.
    subsequence_offsets[0] = 0.; //AsUniform01(hash2(ReverseBits(id)));
    subsequence_offsets[1] = 0.; //AsUniform01(hash3(ReverseBits(id)));
  }
  
  double Uniform01() override
  {
    // I want another sequence. Not the one from the 2d sample.
    // So I xor with scrambling mask with a random number.
    // Probably not sufficient decorelation. Might have to try something more elaborate.
    auto x = RI_vdC(point_idx, scrambling_mask ^ 0x9536a63a);
    x += rotation_offsets[rotation_idx][0];
    x = x >= 1. ? (x - 1.) : x;
    return x;
  }
  
  Double2 UniformUnitSquare() override
  {
    // The first two components of the Sobol sequence.
    // XOR scrambled.
    auto x = RI_vdC(point_idx, scrambling_mask);
    auto y = RI_S(point_idx, scrambling_mask);
    // And Cranley rotated
    x += rotation_offsets[rotation_idx][0];
    y += rotation_offsets[rotation_idx][1];
    x = x >= 1. ? (x - 1.) : x;
    y = (y >= 1.) ? (y - 1.) : y;
    x += subsequence_offsets[0];
    y += subsequence_offsets[1];
    x = x >= 1. ? (x - 1.) : x;
    y = (y >= 1.) ? (y - 1.) : y;
    assert(x >= 0. && x < 1.);
    assert(y >= 0. && y < 1.);
    return { x, y };
  }
};



class PseudoRandomSequence : public SampleSequence
{
  RandGen rg;
public:
  double Uniform01() override { return rg.Uniform01(); }
  Double2 UniformUnitSquare() override { return rg.UniformUnitSquare(); }
};




Sampler::Sampler(bool use_qmc_sequence)
{
  if (use_qmc_sequence)
  {
    sequence = std::make_unique<QuasiRandomSequence>();
  }
  else
  {
    sequence = std::make_unique<PseudoRandomSequence>();
  }
}


RandGen::RandGen()
{
}


void RandGen::Seed(std::uint64_t seed)
{
  generator = pcg32{seed};
}


void RandGen::Uniform01(double* dest, int count)
{
  for (int i=0; i<count; ++i)
  {
    dest[i] = generator.nextDouble();
    assert(std::isfinite(dest[i]));
  }
}


int RandGen::UniformInt(int a, int b_inclusive)
{
  const std::uint32_t x = generator.nextUInt(b_inclusive+1 - a);
  const int y = a + (int)x;
  assert(y >= a && y <= b_inclusive);
  return y;
}

constexpr std::uint64_t RandGen::default_seed;

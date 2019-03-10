#pragma once

#include <cmath>
#include "vec3f.hxx"


class CubeMap
{
  // Use a cube mapping of 6 uniform grids to the 6 side faces of a cube projected to the unit sphere.
  const int bins_per_axis;
  
  // (Mostly) auto generated by my notebook IntegrationOnManifoldsSympy.
  double cube_map_J(double u, double v) const
  {
    using std::pow;
    using std::sqrt;
    double cube_map_J_result;
    cube_map_J_result = sqrt((-pow(u, 2)*pow(v, 2)*pow(pow(u, 2) + pow(v, 2) + 1.0, 2) + (pow(u, 2)*pow(v, 2) + pow(u, 2) + pow(pow(v, 2) + 1.0, 2))*(pow(u, 2)*pow(v, 2) + pow(v, 2) + pow(pow(u, 2) + 1.0, 2)))/pow(pow(u, 2) + pow(v, 2) + 1.0, 6));
    return cube_map_J_result;
  }
  
public:
  CubeMap(int _bins_per_axis) :
    bins_per_axis{_bins_per_axis}
  {}
  
  int TotalNumBins() const
  {
    return bins_per_axis*bins_per_axis*6;
  }
  
  // i,j indices of the grid vertices.
  Double2 VertexToUV(int i, int j) const
  {
    assert (i >= 0 && i <= bins_per_axis);
    assert (j >= 0 && j <= bins_per_axis);
    double delta = 1./bins_per_axis;
    double u = 2.*i*delta-1.;
    double v = 2.*j*delta-1.;
    return Double2{u, v};
  }
  
  // Normally, like here, i,j are indices of the grid cells.
  std::tuple<Double2, Double2> CellToUVBounds(int i, int j) const
  {
    return std::make_tuple(
      VertexToUV(i, j),
      VertexToUV(i+1, j+1));
  }
  
  Double3 UVToOmega(int side, Double2 uv) const
  {
    double u = uv[0];
    double v=  uv[1];
    Double3 x;
    switch (side)
    {
      case 0:
        x = Double3{1., u, v}; break;
      case 1:
        x = Double3{-1., u, v}; break;
      case 2:
        x = Double3{u, 1., v}; break;
      case 3:
        x = Double3{u, -1., v}; break;
      case 4:
        x = Double3{u, v, 1.}; break;
      case 5:
        x = Double3{u, v, -1.}; break;
      default:
        assert(!"We should not get here.");
    }
    return Normalized(x);
  }
  
  double UVtoJ(Double2 uv) const
  {
    return cube_map_J(uv[0], uv[1]);
  }
  
  std::tuple<int, int, int> OmegaToCell(const Double3 &w) const
  {
    double u = 0, v = 0, z = 0;
    int max_abs_axis;
    z = w.array().abs().maxCoeff(&max_abs_axis);
    int side = max_abs_axis*2 + (w[max_abs_axis]>0 ? 0 : 1);
    switch(side)
    {
      case 0:
        u = w[1]; v = w[2]; z = w[0]; break;
      case 1:
        u = w[1]; v = w[2]; z = -w[0]; break;
      case 2:
        u = w[0]; v = w[2]; z = w[1]; break;
      case 3:
        u = w[0]; v = w[2]; z = -w[1]; break;
      case 4:
        u = w[0]; v = w[1]; z = w[2]; break;
      case 5:
        u = w[0]; v = w[1]; z = -w[2]; break;
      default:
        assert(!"Should not get here!");
    }
    assert(z > 0.);
    u /= z;
    v /= z;
    assert(u >= -1.001 && u <= 1.001);
    assert(v >= -1.001 && v <= 1.001);
    int i = (u+1.)*0.5*bins_per_axis;
    int j = (v+1.)*0.5*bins_per_axis;
    i = std::max(0, std::min(bins_per_axis-1, i));
    j = std::max(0, std::min(bins_per_axis-1, j));
    return std::make_tuple(side, i, j);
  }
  
  int CellToIndex(int side, int i, int j) const
  {
    assert (i >= 0 && i < bins_per_axis);
    assert (j >= 0 && j < bins_per_axis);
    assert (side >= 0 && side < 6);
    const int num_per_side = bins_per_axis*bins_per_axis;
    return side*num_per_side + i*bins_per_axis + j;
  }
  
  std::tuple<int, int, int> IndexToCell(int idx) const
  {
    const int num_per_side = bins_per_axis*bins_per_axis;
    assert (idx >= 0 && idx < num_per_side*6);
    int side = idx / num_per_side;
    idx %= num_per_side;
    int i = idx / bins_per_axis;
    idx %= bins_per_axis;
    int j = idx;
    assert (i >= 0 && i < bins_per_axis);
    assert (j >= 0 && j < bins_per_axis);
    assert (side >= 0 && side < 6);
    return std::make_tuple(side, i, j);
  }
};



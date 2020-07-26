#pragma once

#include "eigen.hxx"
#include "util.hxx"

template<int dim_, class Index_ = int>
class LatticeIndexing // Uniform lattice -> integer type index or pointer
{
public:
  using Index = Index_;
  static constexpr int dim = dim_;
  using IVec = Eigen::Array<Index, dim, 1>;

private:
  IVec size;
  IVec strides;

  // see http://eigen.tuxfamily.org/dox-devel/TopicTemplateKeyword.html
  Index dotStrides(const IVec &x) const
  {
    return strides.matrix().dot(x.matrix());
  }

  static IVec CalcStrides(const IVec &l)
  {
    IVec r;
    r[0] = 1;
    for (int i = 1; i < dim; ++i)
      r[i] = l[i - 1] * r[i - 1];
    return r;
  }


public:
  const IVec& Size() const { return size; }
  Index NumSites() const { return strides[dim - 1] * size[dim - 1]; }
  const IVec& Strides() const { return strides; }


  LatticeIndexing(const IVec &l_ = IVec::Zero())
    : size{ l_ }, strides{ CalcStrides(l_) }
  {
  }

  Index LatticeToSite(const IVec &p) const
  {
    // see http://eigen.tuxfamily.org/dox-devel/TopicTemplateKeyword.html
    return dotStrides(p);
  }

  IVec SiteToLattice(Index site) const
  {
    IVec r;
    for (int i = dim - 1; i >= 1; --i)
    {
      r[i] = (int)(site / strides[i]);
      site %= strides[i];
    }
    r[0] = (int)(site / strides[0]);
    return r;
  }
};



/* Manages a contiguous memory block organized as n-dimensional array.
   First dimension moves fastest in linear memory. No shared memory or
   reference counting.
*/
template<class T, int dim_, class Allocator_ = util::AlignedAllocator<T, 16>>
class SimpleNdArray
{
public:
  static constexpr int dim = dim_;
  using Index = int;
  using Lattice = LatticeIndexing<dim, Index>;
  using IVec = typename Lattice::IVec;

private:
  using Allocator = typename Allocator_::template rebind<T>::other;
  Lattice lattice_; // set it up once and never change it. Its number of sites is used to destruct array elements
  T* allocation_pointer;
  Allocator allocator;

public:
  SimpleNdArray() : 
    allocation_pointer{nullptr}
  {
  }

  SimpleNdArray(const IVec &size)
    : lattice_{size}
  {
    
    const std::size_t count = lattice_.NumSites();
    allocation_pointer = allocator.allocate(count);
    std::uninitialized_default_construct_n(allocation_pointer, count);
  }

  ~SimpleNdArray()
  {
    if (allocation_pointer)
    {
      const std::size_t count = lattice_.NumSites();
      std::destroy_n(allocation_pointer, count);
      allocator.deallocate(allocation_pointer, count);
    }
  }

  friend void swap(SimpleNdArray& a, SimpleNdArray& b)
  {
    std::swap(a.lattice_, b.lattice_);
    std::swap(a.allocation_pointer, b.allocation_pointer);
    std::swap(a.allocator, b.allocator);
  }

  SimpleNdArray(SimpleNdArray &&other)
    : lattice_(other.lattice_), 
      allocation_pointer(other.allocation_pointer),
      allocator(std::move(other.allocator))
  {
    other.lattice_ = Lattice{};
    other.allocation_pointer = nullptr;
  }

  SimpleNdArray& operator=(SimpleNdArray &&other)
  {
    swap(*this, other);
    return *this;
  }

  SimpleNdArray(const SimpleNdArray &other)
    : lattice_(other.lattice_)
  {
    const std::size_t count = lattice_.NumSites();
    allocation_pointer = allocator.allocate(count);
    std::uninitialized_copy_n(other.allocation_pointer, count, allocation_pointer);
  }

  SimpleNdArray& operator=(const SimpleNdArray &other)
  {
    SimpleNdArray tmp{other};
    swap(*this, tmp);
    return *this;
  }

  const IVec& shape() const
  {
    return lattice_.Size();
  }

  int size() const 
  {
    return lattice_.NumSites();
  }

  const Lattice& lattice() const
  {
    return lattice_;
  }

  T& operator[](const IVec &index)
  {
    auto p = lattice_.LatticeToSite(index);
    assert(p >= 0 && p < size());
    return allocation_pointer[p];
  }

  const T& operator[](const IVec &index) const
  {
    auto p = lattice_.LatticeToSite(index);
    assert(p >= 0 && p < size());
    return allocation_pointer[p];
  }

  T* begin()
  {
    return data();
  }

  T* end()
  {
    return data()+size();
  }

  T* data()
  {
    return allocation_pointer;
  }

  const T* data() const
  {
    return allocation_pointer;
  }
};


template<class T, int dim_, class Allocator_ = util::AlignedAllocator<T, 16>>
class SimpleLookupTable
{
public:
  static constexpr int dim = dim_;

  Eigen::Array<double, dim,1> origin{ Eigen::zero };
  Eigen::Array<double, dim,1> inv_cell_size{ Eigen::ones };
  SimpleNdArray<T, dim, Allocator_> data;

  const T& operator()(const Eigen::Array<double, dim, 1> &coords) const
  {
    auto index = ((coords - origin) * inv_cell_size).template cast<int>().eval();
    index = index.max(0).min(data.shape() - 1);
    return data[index];
  }
};
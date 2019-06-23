#pragma once

#include "util.hxx"
#include "vec3f.hxx"

/* Adapted from https://github.com/SmallVCM/SmallVCM/blob/master/src/hashgrid.hxx.
 */
class HashGrid 
{
    int cell_count;
    ToyVector<int> cell_starts;
    ToyVector<int> cell_data;
    double inv_cell_size;
    
public:
    HashGrid(double search_radius, const ToyVector<Double3> &items)
    {
        /* Algorithm for building:
         *   Count points for each cell.
         *   Compute start indices
         *   For each point:
         *      set index in cell data array
         *      increase cell end pointer
        */
        double cell_size = 2.f * search_radius;
        inv_cell_size = 1.f / cell_size;

        const int item_count = isize(items);

        cell_count = item_count + 1;  // +1 Prevents taking modulo 0 in GetCellIndex if there are no items.
        std::vector<int> cell_counts(cell_count, 0);
        for (const auto &p : items)
        {
            int cell = GetCellIndex(p);
            cell_counts[cell]++;
        }
        // Now cell_ends contains the number of items in each cell.

        // Now compute cell start indices
        cell_starts.reserve(cell_count+1);
        int counter = 0;
        for (int i=0; i< cell_count; ++i)
        {
            cell_starts.push_back(counter);
            counter += cell_counts[i];
        }
        cell_starts.push_back(counter);
        assert(counter == item_count);

        // Clear cell counts for reuse
        std::fill(cell_counts.begin(), cell_counts.end(), 0);

        // Insert data
        cell_data.resize(item_count, 0);
        int item_idx = 0;
        for (const auto &p : items)
        {
            int cell = GetCellIndex(p);
            int loc = cell_starts[cell] + cell_counts[cell];
            cell_counts[cell]++;
            cell_data[loc] = item_idx++;
        }
    }

    inline double Radius() const { return 0.5/inv_cell_size; }
    
    // Iterates over the 8 closest buckets to the given point.
    // This yields most likely more points than the ones we are interested in.
    // Therefore, the user must additionally filter the returned points
    // by their distance to the query point.
    template<class Visitor>
    void Query(const Double3 &point, Visitor &&visitor)
    {
        decltype(point) scaled_point = inv_cell_size * point;
        int x = (int)std::floor(scaled_point[0]);
        int y = (int)std::floor(scaled_point[1]);
        int z = (int)std::floor(scaled_point[2]);
        auto fx = scaled_point[0] - x;
        auto fy = scaled_point[1] - y;
        auto fz = scaled_point[2] - z;
        for (int i=0; i<8; ++i)
        {
            int cx = ((i & 1) != 0) ? x : (x + (fx < 0.5 ? -1 : 1));
            int cy = ((i & 2) != 0) ? y : (y + (fy < 0.5 ? -1 : 1));
            int cz = ((i & 4) != 0) ? z : (z + (fz < 0.5 ? -1 : 1));
            int cell = GetCellIndex(cx, cy, cz);
            for (int j=cell_starts[cell]; j<cell_starts[cell+1]; ++j)
            {
                visitor(cell_data[j]);
            }
        }
    }
    
private:
    int GetCellIndex(int x, int y, int z)
    {
        // Computes hash and from hash the bucket index in one go. Just like in a normal hash table.
        auto ux = (unsigned)(x);
		auto uy = (unsigned)(y);
		auto uz = (unsigned)(z);
        return (int) (((ux* 73856093) ^ (uy* 19349663) ^
            (uz* 83492791)) % (unsigned) (cell_count));
    }

    int GetCellIndex(const Double3 &point)
    {
        // std::floor to get the next smallest integer, also for negative numbers. E.g. -2.3 becomes 3.0. Cast to int would yield 2.
        int x = (int)std::floor(point[0] * inv_cell_size);
        int y = (int)std::floor(point[1] * inv_cell_size);
        int z = (int)std::floor(point[2] * inv_cell_size);
        return GetCellIndex(x, y, z);
    }
};

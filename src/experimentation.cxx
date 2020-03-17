
#include "eigen.hxx"

#include <iostream>

int main()
{
    static constexpr int N = 1024;
    char buffer[N];
    for (int i=0; i<N; ++i)
        buffer[i] = 'a';
    
    auto* x = new (buffer) Eigen::Array<float,6,1>{Eigen::zero};
    std::cout << *x << std::endl;
}
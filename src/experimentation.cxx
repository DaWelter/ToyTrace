
#include "eigen.hxx"
#include "pcg32/pcg32.h"

#include <iostream>

int main()
{
    pcg32 gen;
    // for (int i=0; i<100; ++i)
    // {
    //     for (int j=0; j<10; ++j)
    //     {
    //         std::cout << gen.nextDouble() << ", ";
    //     }
    //     std::cout << "\n";
    // }
    for (long i=0; ; ++i)
    {
        double d = gen.nextDouble();
        if (d < 1.e-31)
        {
            std::cout << "i = " << i << std::endl;
        }
        if (i % 100000)
            std::cout << ".";
    }
    return 0;
}
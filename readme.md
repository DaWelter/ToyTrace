ToyTrace
========

[Picz](https://www.dropbox.com/sh/vevib9qe5r87a24/AACuqKUPGzxHyl6E2E7iepSha?dl=0)


Features
--------
* Forward Path Tracing
* Bidirectional Path Tracing
* Participating Media:
    * Homogeneous chromatic collision coefficients
    * Atmosphere model with tabulated altitude profiles
    * Uniform-, Henley-Greenstein- and Rayleigh phase functions
    * Spectral delta tracking variant by Kutz et al.
* Materials:
    * Glossy microfacet metallic BRDF
    * Dense dielectric with specular interface
    * Specular mirror
    * Pure diffuse
* Textures & UV mapping
* Lights:
    * Isotropic homogeneous area
    * Parallel area
    * Infinitely distant, including a sun model with realistic spectrum
    * Isotropic sky dome
    * Point
* Camera:
    * Pinhole
    * Fisheye
* Naive Kd-tree
* Binned spectral representation
* Multithreading

Not Implemented
---------------
Eventually, maybe ...

* Advanced techniques. Beam radiance estimators or MLT.
* Refractive interfaces
* SAH based Kd-tree or BVH construction
* Designed for data locality and cache performance
* Ray packet traversal and coherence improving ray-reordering
* Robust intersection computations in the style of BPRT.
* Or just using Intel's Embree
* Glossy dielectric
* Inhomogeneous media apart from atmospheric models
* Faster MIS weight computation from the VCM paper
* Better image reconstruction filter
* Image based lighting
* Bloom filter
* Physical camera
* Texture filtering
* Bump/normal mapping

Related work
------------
Other hobby projects by which this one is inspired.

* Sky Render by Peter Kutz 
* Photorealizer by Peter Kutz
* Takua Render by Yining Karl Li
* Tungsten Renderer by Benedikt Bitterli
* Mitsuba Render by Wenzel Jakob
* Nori an educational ray tracer, by Wenzel Jakob and Steve Marschner
* MagicaVoxel by ephtracy
* smallpt "Global Illumination in 99 lines of C++" by Kevin Beason

There is another ToyTrace on Github. I have nothing to do with it. I just happen to forget to check if the name is already assigned.

License
-------

All rights reserved.

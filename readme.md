ToyTrace
========
Born out of an ancient computer graphics exercise, this is my hobby renderer.

[Picz](https://www.dropbox.com/sh/vevib9qe5r87a24/AACuqKUPGzxHyl6E2E7iepSha?dl=0)

This is experimental software. It works on my computer. It might not on yours.

Dependencies:

* CIMG
* Eigen 3
* boost
* assimp

Builds on Ubuntu 16 with default gcc. Maybe it compiles with VC, but you'd need to have the dependencies as well ofc.

Features
--------
* Forward path tracing with next event estimation
* Bidirectional path tracing
* Participating media:
    * Homogeneous chromatic collision coefficients
    * Atmosphere model with tabulated altitude profiles
    * Uniform-, Henley-Greenstein- and Rayleigh phase functions
    * Spectral delta tracking
* Materials:
    * Glossy microfacet metallic BRDF
    * Dense dielectric with specular interface
    * Specular mirror
    * Specular transmissive dielectric, aka. glass.
    * Lambertian diffuse
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
* Statistical test for BSDF & Phasefunction sampling routines.
* Naive Kd-tree
* Binned spectral representation
* Multithreading
* Automated rendering of test scenes.

Not Implemented
---------------
Eventually, maybe ...

* Advanced techniques. Beam radiance estimators or MLT.
* SAH based Kd-tree or BVH construction
* Designed for data locality and cache performance
* Ray packet traversal and coherence improving ray-reordering
* Robust intersection computations in the style of BPRT.
* Or just using Intel's Embree
* Glossy dielectric
* Inhomogeneous media apart from atmospheric models
* Faster MIS weight computation from the VCM paper
* Better image reconstruction filter
* Firefly suppression
* Stratified/quasi-random sampling
* Image based lighting
* Bloom filter
* Physical camera
* Texture filtering
* Bump/normal mapping

Related work
------------
Other projects by which this one is inspired.

* Sky Render by Peter Kutz (http://skyrenderer.blogspot.de/)
* Photorealizer by Peter Kutz (http://photorealizer.com/)
* Takua Render by Yining Karl Li (https://www.yiningkarlli.com/projects/takuarender.html)
* Tungsten Renderer by Benedikt Bitterli (https://benedikt-bitterli.me/tungsten.html)
* MagicaVoxel by ephtracy (https://ephtracy.github.io/)
* smallpt "Global Illumination in 99 lines of C++" by Kevin Beason (http://www.kevinbeason.com/smallpt/)
* JavaScript Path Tracer by Hunter Loftis  (https://github.com/hunterloftis/pathtracer)
* pbr: a Physically-Based 3D Renderer in Go by Hunter Loftis (https://github.com/hunterloftis/pbr)
* Path Tracer by Michael Fogleman (https://www.michaelfogleman.com)

Educational/Research/Reference Implementations

* Nori an educational ray tracer (http://www.cs.cornell.edu/courses/cs6630/2012sp/nori/, https://wjakob.github.io/nori/)
* PBRT (https://github.com/mmp)
* SmallVCM (http://www.smallvcm.com/)
* Unified points, beams and paths (https://github.com/PetrVevoda/smallupbp)
* Lightmetrica (http://lightmetrica.org/)
* Mitsuba Render (https://www.mitsuba-renderer.org/)

There is another ToyTrace on Github. I have nothing to do with it. I just happen to forget to check if the name is already assigned.

License
-------

All rights reserved.

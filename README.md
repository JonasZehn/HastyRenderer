<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template and https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

# Hasty Renderer

![Hasty teaser](images/teaser.jpg)


<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">Hasty Renderer</h3>

  <p align="center">
    An offline path tracer hastily coded
    <br />
    <br />
    <a href="#gallery">View Gallery</a>
    ·
    <a href="https://github.com/JonasZehn/HastyRenderer/issues">Report Bug</a>
    ·
    <a href="https://github.com/JonasZehn/HastyRenderer/issues">Request Feature</a>
  </p>
</div>


## About the project

I'm throwing together this pathtracer to learn more details about path tracing, the [Meson](https://mesonbuild.com/) build system and modern C++ features. 
This project currently includes a unidirectional path tracer and a stochastic progressive photon mapper.
[Intel Embree](https://www.embree.org/) is used for tracing rays and [Intel Open Image Denoise](https://www.openimagedenoise.org/) is used for denoising.


## Usage

### Requirements
* Windows
* [Meson](https://mesonbuild.com/) or [CMake](https://cmake.org/)
* C++17 Compiler

### Build with Meson

First, setup the variables in `meson_vars.bat`.
Then, there are batch files included to run the necessary build commands which are of the form of

```
meson setup %builddir% --backend %backend% --buildtype=%buildtype% -DZLIB_INCLUDE_DIR=%ZLIB_INCLUDE_DIR% -DZLIB_LIBRARY=%ZLIB_LIBRARY%
meson compile -C %builddir%

```
Most dependencies are automatically fetched by Meson, except for zlib. When using CMake, the d

### Build with CMake
First, setup the variables in `cmake_vars.bat`. Install dependencies using [vcpkg](https://vcpkg.io/) by running the `vcpkg_install_requirements_windows.bat` file. Then run the `cmake_build.bat` file to configure and build the project.

### Input format

Currently the input format is quite simple, i.e. it the renderer takes two JSON files and an .obj file which can be e.g. exported from blender. Of course, not all features are preserved due to the choice of the file format, which I plan to change eventually.

## Roadmap

* [x] Unidirectional Path Tracer
* [x] Denoiser Integration
* [x] "Physically based" Materials
* [x] Stochastic Progressive Photon Mapper
* [x] Basic Textures
* [x] Interpolated Normals
* [x] Depth of Field
* [x] Environment Map
* [x] Aperture Blades
* [x] CMake Build
* [ ] Normal Map
* [ ] Hero Wavelength
* [ ] Microfacet based Refraction
* [ ] Path Guiding integration
* [ ] USD
* [ ] Linux/MacOS Build

## References

- [An Overview of BRDF Models, 2012, R. A. Montes Soldado, C. Ureña Almagro](https://digibug.ugr.es/handle/10481/19751)
- [Hero Wavelength Spectral Sampling, 2014, A. Wilkie, S. Nawaz, M. Droske, A. Weidlich, J. Hanika](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.12419)
- [Importance Sampling techniques for GGX with Smith Masking-Shadowing: Part 2, 2018, J. Schutte](https://schuttejoe.github.io/post/ggximportancesamplingpart2/)
- [Non-symmetric scattering in light transport algorithms, 1996, E. Veach](https://link.springer.com/chapter/10.1007/978-3-7091-7484-5_9)
- [Realistic Image Synthesis Using Photon Mapping, 2001, H. W. Jensen](https://gitea.yiem.net/QianMo/Real-Time-Rendering-4th-Bibliography-Collection/raw/branch/main/Chapter%201-24/[0822]%20[Book%202001]%20Realistic%20Image%20Synthesis%20Using%20Photon%20Mapping.pdf)
- [Robust Monte Carlo methods for light transport simulation, 1998, E. Veach](http://graphics.stanford.edu/papers/veach_thesis/thesis.pdf)
- [Stochastic Progressive Photon Mapping, 2009, T. Hachisuka, H. W. Jensen](https://dl.acm.org/doi/abs/10.1145/1661412.1618487)
- [Physically Based Shading at Disney, 2012, B. Burley](https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf)

 ## Useful Resources
 - [McGuire Computer Graphics Archive - Meshes](https://casual-effects.com/data/)
 - [Polyhaven](https://polyhaven.com/)
 - [AmbientCG](https://ambientcg.com/)
 
 
## Gallery 


|  |  |
| ----------- | ----------- |
| ![glass](images/glass.jpg) | ![cuboids](images/cuboids.jpg) |

| Physically based Materials | RGB Dispersion |
| ----------- | ----------- |
| ![physically based materials](images/physically_based_materials.jpg) | ![rgb dispersion](images/rgb_dispersion.jpg) | 


| Face Normals | Interpolated Normal |
| ----------- | ----------- |
| ![face normals](images/face_normals.jpg) | ![face normals](images/interpolated_normals.jpg) | 


| No MIS | MIS |
| ----------- | ----------- |
| ![Output](images/no_mis.jpg) | ![Output](images/mis.jpg) | 

| Raw - 10 Samples per Pixel  | Denoised (Intel® Open Image Denoise)  |
| ----------- | ----------- |
| ![Output](images/not_denoised.jpg) | ![Output](images/denoised.jpg) | 

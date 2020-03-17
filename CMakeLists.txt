cmake_minimum_required(VERSION 3.5)
project(toytrace)

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set (Rapidjson_ENABLE False CACHE BOOL "For debug data output and loading of atmospheric data.")
set (BuildTests True CACHE BOOL "Build the tests")
set (BuildPythonBindings True CACHE BOOL "Build Python bindings")

if(NOT WIN32)
    set (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++17) # Set this only for c++ files.
    add_compile_options (-Wextra -Wall)
    add_compile_options (-fPIC)
    add_compile_options (-Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-char-subscripts -Winit-self -Wno-unused-function)
    add_compile_options ("$<$<CONFIG:RELEASE>:-march=native>")
    add_compile_options ("$<$<CONFIG:RELEASE>:-O3>")
    add_compile_options ("$<$<CONFIG:RELWITHDEBINFO>:-march=native>")
    add_compile_options ("$<$<CONFIG:RELWITHDEBINFO>:-fno-omit-frame-pointer>")
    #add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    #set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address)
else()
    add_compile_options(-std:c++17)
    add_definitions(-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -D_NOEXCEPT=noexcept)
    add_definitions(-D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING) # Mostly for YAML-Cpp
    add_definitions(-D_SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING) # For Eigen
    add_definitions(-DNOGDI -DWIN32_LEAN_AND_MEAN) # Don't let windows api pollute the global namespace with silly macros. windows.h is included by TBB. Nothing else I can do about that.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4305")  # float <-> double conversion. Until I get that mess sorted out.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")  # Unkown pragma
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4099")  # pdb file not found
endif()
add_compile_options ("$<$<CONFIG:DEBUG>:-DEIGEN_INITIALIZE_MATRICES_BY_NAN>")

# Eliminate link error caused by boost/pool/simple_segregated_storage.hpp
# https://stackoverflow.com/questions/9723793/undefined-reference-to-boostsystemsystem-category-when-compiling
add_definitions("-DBOOST_SYSTEM_NO_DEPRECATED")

# Because bug. Need to fix ...
# add_definitions("-DEIGEN_DONT_VECTORIZE")


#################################################
### Dependencies
#################################################



# Just to get TBB to work ...
if (WIN32)
    find_package(tbb REQUIRED)
    # VCPKG stuff. Any way to make it independent of package manager?
    # Regular find script cannot find it. On Ubuntu, the config file which
    # imports this nice TBB::tbb target is not available.
    # So I have to get to a common ground here.
    #get_target_property(TBB_DEBUG_LIBRARIES TBB::tbb IMPORTED_LOCATION_DEBUG)
    #get_target_property(TBB_LIBRARIES TBB::tbb IMPORTED_LOCATION_RELEASE)
    #get_target_property(TBB_INCLUDE_DIRS TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
else()
    find_package(TBB REQUIRED)
    if (NOT TBB_DEBUG_LIBRARIES)
        # On linux there are no special debug libs? But we need to set it to be able
        # to deal use the same link command on windows and linux.
        set (TBB_DEBUG_LIBRARIES ${TBB_LIBRARIES})
    endif()
endif(WIN32)
#message(WARNING "Tbb libs are: ${TBB_LIBRARIES} ${TBB_DEBUG_LIBRARIES} and the include dirs: ${TBB_INCLUDE_DIRS}")
find_package(X11)
find_package(Threads REQUIRED)
find_package(assimp REQUIRED)
find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options filesystem system)
find_path(CImg_INCLUDE_DIR
  NAMES CImg.h
  PATHS
    ${CMAKE_INSTALL_PREFIX}/include
    /usr/include
    /usr/local/include
)
message(STATUS "CImg: ${CImg_INCLUDE_DIR}")
find_path(Embree_INCLUDE_DIR
  NAMES rtcore.h
  PATHS
    ${CMAKE_INSTALL_PREFIX}/include
    /usr/include
    /usr/local/include)
find_library(Embree_LIBRARY NAMES libembree3)
message(STATUS "Embree: ${Embree_INCLUDE_DIR}, ${Embree_LIBRARY}")
find_library(OPENIMAGEIO_LIBRARY NAMES openimageio OpenImageIO)

find_package(JPEG)
find_package(PNG)

if (Rapidjson_ENABLE)
  find_path(Rapidjson_INCLUDE_DIR
    NAMES rapidjson/rapidjson.h
    PATHS
      ${CMAKE_INSTALL_PREFIX}/include
      /usr/include
      /usr/local/include
  )
  if (NOT Rapidjson_INCLUDE_DIR)
    message(FATAL_ERROR "Rapidjson not found!")
  else()
    message(STATUS "Rapidjson: ${Rapidjson_INCLUDE_DIR}")
  endif()
  add_definitions("-DHAVE_JSON")
endif()

find_path(YAML_CPP_INCLUDE_DIR 
  NAMES yaml-cpp/yaml.h
  PATHS
    ${CMAKE_INSTALL_PREFIX}/include
    /usr/include
    /usr/local/include
)
find_library(YAML_CPP_LIBRARY NAMES yaml-cpp PATHS /usr/local/lib)

message(STATUS "Yaml-cpp Inlcude Path: ${YAML_CPP_INCLUDE_DIR}")
message(STATUS "Yaml-cpp LIBRARY: ${YAML_CPP_LIBRARY}")


#################################################
### Library with common stuff 
#################################################

add_library(eigen_plugins INTERFACE)
target_include_directories(eigen_plugins INTERFACE "src/eigen_plugins")


add_library(commonstuff STATIC
  src/image.cxx  src/parsenff.cxx  src/shader.cxx src/ray.cxx src/sampler.cxx
  src/phasefunctions.cxx src/atmosphere.cxx src/spectral.cxx src/primitive.cxx src/renderingalgorithms.cxx 
  src/light.cxx src/texture.cxx src/embreeaccelerator.cxx src/scene.cxx src/lightpicker_trivial.cxx src/lightpicker_ucb.cxx src/renderingalgorithms_photonmap.cxx src/renderingalgorithms_pathtracing2.cxx 
  src/renderingalgorithms_normalvisualizer.cxx src/renderingalgorithms_pathtracing.cxx
  src/rendering_util.cxx src/pathlogger.cxx src/photonintersector.cxx  external/cubature/hcubature.c 
  src/path_guiding.cxx src/renderingalgorithms_pathtracing_guided.cxx src/distribution_mixture_models.cxx src/path_guiding_tree.cxx)

  file(GLOB headers RELATIVE ${CMAKE_SOURCE_DIR} "src/*.hxx")
target_sources(commonstuff PRIVATE ${headers})

target_link_libraries(commonstuff 
    ${ASSIMP_LIBRARIES} 
    ${Embree_LIBRARY} 
    ${OPENIMAGEIO_LIBRARY} 
    optimized ${Boost_FILESYSTEM_LIBRARY_RELEASE} 
    debug ${Boost_FILESYSTEM_LIBRARY_DEBUG} 
    optimized ${Boost_SYSTEM_LIBRARY_RELEASE} 
    debug ${Boost_SYSTEM_LIBRARY_DEBUG}     
    ${PNG_LIBRARY} ${JPEG_LIBRARIES} 
    ${CMAKE_THREAD_LIBS_INIT} 
    ${X11_LIBRARIES} 
    ${YAML_CPP_LIBRARY}
    optimized ${TBB_LIBRARIES}
    debug ${TBB_DEBUG_LIBRARIES}
    eigen_plugins
)

target_include_directories(commonstuff PUBLIC
    ${PNG_INCLUDE_DIR} 
    ${JPEG_INCLUDE_DIR} 
    ${CImg_INCLUDE_DIR} 
    ${Rapidjson_INCLUDE_DIR} 
    ${Embree_INCLUDE_DIR} 
    ${YAML_CPP_INCLUDE_DIR} 
    ${TBB_INCLUDE_DIRS} 
    "external/cubature")


#################################################
### Executables
#################################################

add_executable(toytrace src/toytrace.cxx)
target_link_libraries(toytrace commonstuff 
                      optimized ${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE}
                      debug ${Boost_PROGRAM_OPTIONS_LIBRARY_DEBUG})

if(BuildTests)
    add_executable(tests src/tests.cxx src/tests3.cxx src/tests_sampling.cxx src/tests_microfacet.cxx src/tests_stats.cxx src/tests_scene.cxx src/tests_guiding.cxx)
    add_executable(tests2 src/tests2.cxx)
    add_executable(tests_tbb src/tests_tbb.cxx)
    add_executable(experimentation src/experimentation.cxx)

    foreach(EXE tests tests2 tests_tbb)
      target_include_directories(${EXE} PUBLIC ${GTEST_INCLUDE_DIRS})
      target_link_libraries(${EXE} commonstuff ${GTEST_LIBRARIES})
    endforeach()

    target_link_libraries(experimentation commonstuff)
endif(BuildTests)


#################################################
## Python Lib
#################################################

if (BuildPythonBindings)
  find_package(PythonInterp REQUIRED)
  find_package(PythonLibs REQUIRED)

  add_library(path_guiding SHARED src/path_guiding_py.cxx)
  target_link_libraries(path_guiding commonstuff ${PYTHON_LIBRARIES})
  target_include_directories(path_guiding PRIVATE ${PYTHON_INCLUDE_DIRS})

  set_target_properties(
      path_guiding
      PROPERTIES
          PREFIX ""
          OUTPUT_NAME "path_guiding"
          LINKER_LANGUAGE C
      )
endif(BuildPythonBindings)
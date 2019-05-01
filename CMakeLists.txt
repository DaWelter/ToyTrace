cmake_minimum_required(VERSION 2.8)
project(toytrace)

set (Rapidjson_ENABLE False CACHE BOOL "For debug data output and loading of atmospheric data.")
set (BuildTests True CACHE BOOL "Build the tests")

if(NOT WIN32)
    set (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++17) # Set this only for c++ files.
    add_compile_options (-Wextra -Wall)
    add_compile_options (-Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-char-subscripts -Winit-self -Wno-unused-function)
    add_compile_options ("$<$<CONFIG:RELEASE>:-march=native>")
    add_compile_options ("$<$<CONFIG:RELEASE>:-O3>")
    add_compile_options ("$<$<CONFIG:RELEASE>:-msse2>")
    add_compile_options ("$<$<CONFIG:RELEASE>:-mfpmath=sse>")
    # Eliminate link error caused by boost/pool/simple_segregated_storage.hpp
    # https://stackoverflow.com/questions/9723793/undefined-reference-to-boostsystemsystem-category-when-compiling
    add_definitions("-DBOOST_SYSTEM_NO_DEPRECATED")
    add_definitions("-DEIGEN_DONT_VECTORIZE")

    find_package(X11)
    find_package(Threads REQUIRED)
    find_library(ASSIMP_LIBRARY NAMES assimp)
    find_package(GTest REQUIRED)
    find_package(Boost REQUIRED COMPONENTS program_options filesystem)
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
    message(STATUS "OpenImageIO: ${OPENIMAGEIO_LIBRARY}")

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

    include_directories(${PNG_INCLUDE_DIR} ${JPEG_INCLUDE_DIR} ${CImg_INCLUDE_DIR} ${Rapidjson_INCLUDE_DIR} ${Embree_INCLUDE_DIR} ${YAML_CPP_INCLUDE_DIR})
    link_libraries(${PNG_LIBRARY} ${JPEG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${ASSIMP_LIBRARY} ${X11_LIBRARIES} ${Embree_LIBRARY} ${OPENIMAGEIO_LIBRARY} ${YAML_CPP_LIBRARY})
else()
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    add_compile_options(-std:c++17)
    add_definitions(-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -D_NOEXCEPT=noexcept)
    conan_basic_setup()
endif(NOT WIN32)
    
set(Cubature_DIR "external/cubature")

    
set(SOURCES src/image.cxx  src/parsenff.cxx  src/shader.cxx src/ray.cxx src/sampler.cxx
            src/phasefunctions.cxx src/atmosphere.cxx src/spectral.cxx src/primitive.cxx src/renderingalgorithms.cxx 
            src/light.cxx src/texture.cxx src/embreeaccelerator.cxx src/scene.cxx src/lightpicker_trivial.cxx src/renderingalgorithms_photonmap.cxx 
            src/rendering_util.cxx src/pathlogger.cxx src/photonintersector.cxx)
add_executable(toytrace src/toytrace.cxx ${SOURCES})


if(BuildTests)
    add_executable(tests src/tests.cxx src/tests3.cxx src/tests_sampling.cxx src/tests_microfacet.cxx src/tests_stats.cxx ${Cubature_DIR}/hcubature.c ${SOURCES})
    add_executable(tests2 src/tests2.cxx)
    add_executable(tests_tbb src/tests_tbb.cxx)
    target_link_libraries(tests_tbb tbb)
endif(BuildTests)


if(NOT WIN32)
    target_link_libraries(toytrace ${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE} ${Boost_FILESYSTEM_LIBRARY_RELEASE} ${Boost_SYSTEM_LIBRARY_RELEASE} tbb)
    if (BuildTests)
        foreach(EXE tests tests2)
          target_include_directories(${EXE} PUBLIC ${GTEST_INCLUDE_DIRS} ${Cubature_DIR})
          target_link_libraries(${EXE} ${GTEST_LIBRARIES}  ${Boost_FILESYSTEM_LIBRARY_RELEASE} ${Boost_SYSTEM_LIBRARY_RELEASE} pthread tbb)
        endforeach()
    endif(BuildTests)
else()
    target_link_libraries(toytrace ${CONAN_LIBS})
endif()
    
# # For qtcreator to display all header files. (https://stackoverflow.com/questions/28384935/qtcreator-cmake-project-how-to-show-all-project-files)
# FILE(GLOB THE_HEADERS "src/*.hxx")
# add_custom_target(headers SOURCES ${THE_HEADERS})

# # The same for scenes because I want to open them in the IDE for convenience.
# FILE(GLOB THE_SCENES "scenes/*.nff")
# add_custom_target(scenes SOURCES ${THE_SCENES})

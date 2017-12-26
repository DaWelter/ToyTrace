cmake_minimum_required(VERSION 2.8)
project(toytrace)

add_compile_options (-std=c++14)
add_compile_options (-Wextra -Wall)
add_compile_options (-Wno-sign-compare -Wno-unused-parameter -Wno-reorder -Wno-unused-variable -Wno-char-subscripts)
add_compile_options ("$<$<CONFIG:RELEASE>:-march=native>")
add_compile_options ("$<$<CONFIG:RELEASE>:-O3>")
add_compile_options ("$<$<CONFIG:RELEASE>:-msse2>")
add_compile_options ("$<$<CONFIG:RELEASE>:-mfpmath=sse>")
# Eliminate link error caused by boost/pool/simple_segregated_storage.hpp
# https://stackoverflow.com/questions/9723793/undefined-reference-to-boostsystemsystem-category-when-compiling
add_definitions("-DBOOST_SYSTEM_NO_DEPRECATED")

if(NOT WIN32)
  find_package(X11)
  find_package(Threads REQUIRED)
endif()
find_library(ASSIMP_LIBRARY NAMES assimp)
find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_path(CImg_INCLUDE_DIR
  NAMES CImg.h
  PATHS
    ${CMAKE_INSTALL_PREFIX}/include
    /usr/include
    /usr/local/include
)
message(STATUS "CImg: ${CImg_INCLUDE_DIR}")

find_package(JPEG)
find_package(PNG)

find_path(Rapidjson_INCLUDE_DIR
  NAMES rapidjson/rapidjson.h
  PATHS
    ${CMAKE_INSTALL_PREFIX}/include
    /usr/include
    /usr/local/include
)
message(STATUS "Rapidjson: ${Rapidjson_INCLUDE_DIR}")

include_directories(${PNG_INCLUDE_DIR} ${JPEG_INCLUDE_DIR} ${CImg_INCLUDE_DIR} ${Rapidjson_INCLUDE_DIR})
link_libraries(${PNG_LIBRARY} ${JPEG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${ASSIMP_LIBRARY} ${X11_LIBRARIES})

set(SOURCES image.cxx  parsenff.cxx  shader.cxx ray.cxx sampler.cxx bsptree.cxx phasefunctions.cxx atmosphere.cxx spectral.cxx)
add_executable(toytrace toytrace.cxx ${SOURCES})
target_link_libraries(toytrace ${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE})

add_executable(tests tests.cxx ${SOURCES})
add_executable(tests2 tests2.cxx)
#add_executable(tests3 tests3.cxx)

foreach(EXE tests tests2)
  target_include_directories(${EXE} PUBLIC ${GTEST_INCLUDE_DIRS})
  target_link_libraries(${EXE} ${GTEST_LIBRARIES} pthread)
endforeach()


# For qtcreator to display all header files. (https://stackoverflow.com/questions/28384935/qtcreator-cmake-project-how-to-show-all-project-files)
FILE(GLOB THE_HEADERS "*.hxx")
add_custom_target(headers SOURCES ${THE_HEADERS})

# The same for scenes because I want to open them in the IDE for convenience.
FILE(GLOB THE_SCENES "scenes/*.nff")
add_custom_target(scenes SOURCES ${THE_SCENES})

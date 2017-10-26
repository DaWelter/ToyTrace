cmake_minimum_required(VERSION 2.8)
project(naivetrace)

include_directories(external)

add_compile_options (-std=c++14)
add_compile_options (-Wextra -Wall)
add_compile_options (-Wno-sign-compare -Wno-unused-parameter -Wno-reorder -Wno-unused-variable -Wno-char-subscripts)

find_package(PNG)
find_package(JPEG)
find_package(Threads)
find_library(ASSIMP_LIBRARY NAMES assimp)
find_package(X11)
find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

include_directories(${PNG_INCLUDE_DIR} ${JPEG_INCLUDE_DIR})
link_libraries(${PNG_LIBRARY} ${JPEG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${ASSIMP_LIBRARY} ${X11_LIBRARIES})

set(SOURCES image.cxx  parsenff.cxx  shader.cxx ray.cxx sampler.cxx bsptree.cxx phasefunctions.cxx)
add_executable(naivetrace naivetrace.cxx ${SOURCES})
add_executable(tests tests.cxx ${SOURCES})

target_link_libraries(naivetrace ${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE})

target_include_directories(tests PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(tests ${GTEST_LIBRARIES} pthread)
cmake_minimum_required(VERSION 2.8)
project(toytrace)

add_compile_options (-std=c++14)
add_compile_options (-Wextra -Wall)
add_compile_options (-Wno-sign-compare -Wno-unused-parameter -Wno-reorder -Wno-unused-variable -Wno-char-subscripts)

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
find_package(JPEG)
find_package(PNG)
if (CImg_INCLUDE_DIR)
    message(STATUS "CImg: ${CImg_INCLUDE_DIR}")
endif()

include_directories(${PNG_INCLUDE_DIR} ${JPEG_INCLUDE_DIR} ${CImg_INCLUDE_DIR})
link_libraries(${PNG_LIBRARY} ${JPEG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${ASSIMP_LIBRARY} ${X11_LIBRARIES})

set(SOURCES image.cxx  parsenff.cxx  shader.cxx ray.cxx sampler.cxx bsptree.cxx phasefunctions.cxx atmosphere.cxx)
add_executable(toytrace toytrace.cxx ${SOURCES})
add_executable(tests tests.cxx ${SOURCES})

target_link_libraries(toytrace ${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE})

target_include_directories(tests PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(tests ${GTEST_LIBRARIES} pthread)


# For qtcreator to display all header files. (https://stackoverflow.com/questions/28384935/qtcreator-cmake-project-how-to-show-all-project-files)
FILE(GLOB THE_HEADERS "*.hxx")
add_custom_target(headers SOURCES ${THE_HEADERS})

# The same for scenes because I want to open them in the IDE for convenience.
FILE(GLOB THE_SCENES "scenes/*.nff")
add_custom_target(scenes SOURCES ${THE_SCENES})
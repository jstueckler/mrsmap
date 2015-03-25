cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(mrsmap)

# In case the user does not setup CMAKE_BUILD_TYPE, assume it's RelWithDebInfo
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "build type default to RelWithDebInfo, set to Release to improve performance" FORCE)
endif("${CMAKE_BUILD_TYPE}" STREQUAL "")

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  message(STATUS "building in release mode")
  add_definitions( -msse4.2 -fopenmp )
  add_definitions("-DBOOST_DISABLE_ASSERTS -DEIGEN_NO_DEBUG -DNDEBUG")
  add_definitions (-Wall -fno-strict-aliasing)# -fvisibility=hidden -fvisibility-inlines-hidden)
  add_definitions(-Wall -funroll-loops -momit-leaf-frame-pointer -fomit-frame-pointer -floop-block -ftree-loop-distribution -ftree-loop-linear -floop-interchange -floop-strip-mine -fgcse-lm -fgcse-sm -fsched-spec-load)
endif()

add_definitions( -std=c++0x )

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})


IF( NOT CATKIN_DEVEL_PREFIX STREQUAL "" )

  ## Find catkin macros and libraries
  ## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
  ## is used, also find other catkin packages
  find_package(catkin)

  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES mrsmap mrsslam
    DEPENDS system_lib
  )

ENDIF()


# PCL required
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# OctreeLib required
find_package(OctreeLib REQUIRED)
include_directories(${OCTREELIB_INCLUDE_DIRS})

# GSL required
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
link_directories(${GSL_LIBRARY_DIRS})

# TBB required
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})

# OpenCV required
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("${OpenCV_INCLUDE_DIRS}")

# G2O required
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIR})

find_package( Boost 1.40 COMPONENTS program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )


# For building the CHOLMOD / CSPARSE solvers
find_package(Cholmod)
find_package(BLAS)
find_package(LAPACK)
find_package(CSparse)
include_directories(${CSPARSE_INCLUDE_DIR})


find_package( Boost 1.40 COMPONENTS program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )


include_directories("/usr/include/openni/")

include_directories(${PROJECT_SOURCE_DIR}/include)



#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)



# -- mrsmap lib for map aggregation, registration, and visualization
set( SRC_MRSMAPLIB    
		src/map/multiresolution_surfel_map.cpp
		src/map/nodevalue.cpp
		src/map/surfel.cpp
		src/map/surfelpair.cpp
		src/map/shapetexture_feature.cpp
		src/registration/multiresolution_surfel_registration.cpp
		src/registration/multiresolution_soft_surfel_registration.cpp
		src/registration/multiresolution_surfel_initial_alignment.cpp
		src/utilities/utilities.cpp
		src/visualization/visualization_map.cpp
)

add_library( mrsmap SHARED ${SRC_MRSMAPLIB} )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( mrsmap octreelib ${PCL_LIBRARIES} ${OpenCV_LIBS} ${TBB_LIBRARIES} ${GSL_LIBRARIES} )



# -- mrsslam lib for slam with mrsmaps
set( SRC_MRSSLAMLIB    
		src/map/multiresolution_surfel_map.cpp
		src/map/nodevalue.cpp
		src/map/surfel.cpp
		src/map/surfelpair.cpp
		src/map/shapetexture_feature.cpp
		src/registration/multiresolution_surfel_registration.cpp
		src/registration/multiresolution_soft_surfel_registration.cpp
		src/registration/multiresolution_surfel_initial_alignment.cpp
		src/utilities/utilities.cpp
		src/visualization/visualization_map.cpp
)

add_library( mrsslam SHARED src/slam/slam.cpp src/visualization/visualization_slam.cpp )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( mrsslam mrsmap ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_CSPARSE_EXTENSION} ${G2O_TYPES_SLAM3D} )


# -- evaluate_visual_odometry for evaluation of visual odometry performance on RGBD dataset
add_executable( evaluate_visual_odometry src/tools/evaluation/evaluate_visual_odometry.cpp )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( evaluate_visual_odometry mrsmap ${Boost_PROGRAM_OPTIONS_LIBRARY} )


# -- evaluate_visual_odometry_batch for evaluation of visual odometry performance on multiple RGBD datasets
add_executable( evaluate_visual_odometry_batch src/tools/evaluation/evaluate_visual_odometry_batch.cpp )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( evaluate_visual_odometry_batch mrsmap ${Boost_PROGRAM_OPTIONS_LIBRARY} )


# -- evaluate_slam for evaluation of slam performance on RGBD dataset
add_executable( evaluate_slam src/tools/evaluation/evaluate_slam.cpp )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( evaluate_slam mrsslam ${Boost_PROGRAM_OPTIONS_LIBRARY} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_CSPARSE_EXTENSION} ${G2O_TYPES_SLAM3D} )


cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(MRSMap)

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

#check for OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  if(MSVC90 OR MSVC10)
    if(MSVC90)
      set(OPENMP_DLL VCOMP90)
    elseif(MSVC10)
      set(OPENMP_DLL VCOMP100)
    endif(MSVC90)
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DELAYLOAD:${OPENMP_DLL}D.dll")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DELAYLOAD:${OPENMP_DLL}.dll")
  endif(MSVC)
else(OPENMP_FOUND)
  message (STATUS "OpenMP not found")
endif()

if (MSVC)
  Set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /O2 ${SSE_FLAGS}")
else (MSVC)
  set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS} ${SSE_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g ${SSE_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE " ${CMAKE_CXX_FLAGS} -O3 ${SSE_FLAGS}")
endif (MSVC)

add_definitions( -std=c++0x )

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})


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

find_package( Boost 1.40 COMPONENTS program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )


include_directories("/usr/include/openni/")

include_directories(${PROJECT_SOURCE_DIR}/include)



#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)



# -- mrslib for map aggregation, registration, and visualization
set( SRC_MRSMAPLIB    
		src/map/multiresolution_surfel_map.cpp
		src/map/nodevalue.cpp
		src/map/surfel.cpp
		src/map/surfelpair.cpp
		src/map/shapetexture_feature.cpp
		src/registration/multiresolution_surfel_registration.cpp
#		src/registration/multiresolution_soft_surfel_registration.cpp
		src/utilities/utilities.cpp
		src/visualization/visualization_map.cpp
)

add_library( mrsmap SHARED ${SRC_MRSMAPLIB} )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( mrsmap octreelib ${PCL_LIBRARIES} ${OpenCV_LIBS} ${TBB_LIBRARIES} ${GSL_LIBRARIES} )


# -- evaluate_visual_odometry for evaluation of visual odometry performance on RGBD dataset
add_executable( evaluate_visual_odometry src/tools/evaluation/evaluate_visual_odometry.cpp )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( evaluate_visual_odometry mrsmap ${Boost_PROGRAM_OPTIONS_LIBRARY} )


# -- evaluate_visual_odometry_batch for evaluation of visual odometry performance on multiple RGBD datasets
add_executable( evaluate_visual_odometry_batch src/tools/evaluation/evaluate_visual_odometry_batch.cpp )
add_definitions( ${SSE_FLAGS} )
target_link_libraries( evaluate_visual_odometry_batch mrsmap ${Boost_PROGRAM_OPTIONS_LIBRARY} )




###############################################################################
# Find MRSMap
#
# This sets the following variables:
# MRSMAP_FOUND - True if MRSMap was found.
# MRSMAP_INCLUDE_DIRS - Directories containing the MRSMap include files.
# MRSMAP_DEFINITIONS - Compiler flags for MRSMap.

find_path(MRSMAP_INCLUDE_DIR mrsmap/map/multiresolution_csurfel_map.h
    HINTS "$ENV{MRSMAP_ROOT}" "/usr" "/usr/local")

find_library(MRSMAP_LIBRARY NAMES mrsmap libmrsmap
    HINTS "$ENV{MRSMAP_ROOT}/lib" "/usr/lib" "/usr/local/lib")

find_library(MRSMAP_SLAM_LIBRARY NAMES mrsslam libmrsslam
    HINTS "$ENV{MRSMAP_ROOT}/lib" "/usr/lib" "/usr/local/lib")

find_library(MRSMAP_SMOSLAM_LIBRARY NAMES mrssmoslam libmrssmoslam
    HINTS "$ENV{MRSMAP_ROOT}/lib" "/usr/lib" "/usr/local/lib")


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(MRSMap  DEFAULT_MSG
                                  MRSMAP_LIBRARY MRSMAP_SLAM_LIBRARY MRSMAP_SMOSLAM_LIBRARY MRSMAP_INCLUDE_DIR)

set(MRSMAP_INCLUDE_DIRS ${MRSMAP_INCLUDE_DIR})

mark_as_advanced(MRSMAP_INCLUDE_DIR MRSMAP_LIBRARY MRSMAP_SLAM_LIBRARY MRSMAP_SMOSLAM_LIBRARY)

if(MRSMAP_FOUND)
  message(STATUS "MRSMap found (include: ${MRSMAP_INCLUDE_DIR})")
endif(MRSMAP_FOUND)


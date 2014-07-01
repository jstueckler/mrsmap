###############################################################################
# Find OctreeLib
#
# This sets the following variables:
# OCTREELIB_FOUND - True if OctreeLib was found.
# OCTREELIB_INCLUDE_DIRS - Directories containing the OctreeLib include files.
# OCTREELIB_DEFINITIONS - Compiler flags for OctreeLib.

find_path(OCTREELIB_INCLUDE_DIR octreelib/spatialaggregate/octree.h
    HINTS "$ENV{OCTREELIB_ROOT}" "/usr" "/usr/local")

find_library(OCTREELIB_LIBRARY NAMES octreelib liboctreelib
    HINTS "$ENV{OCTREELIB_ROOT}/lib" "/usr/lib" "/usr/local/lib")

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(OctreeLib  DEFAULT_MSG
                                  OCTREELIB_LIBRARY OCTREELIB_INCLUDE_DIR)

set(OCTREELIB_INCLUDE_DIRS ${OCTREELIB_INCLUDE_DIR})

mark_as_advanced(OCTREELIB_INCLUDE_DIR OCTREELIB_LIBRARY)

if(OCTREELIB_FOUND)
  message(STATUS "OctreeLib found (include: ${OCTREELIB_INCLUDE_DIR})")
endif(OCTREELIB_FOUND)


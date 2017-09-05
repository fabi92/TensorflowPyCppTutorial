cmake_minimum_required (VERSION 3.0)

# nvcc GCC 5.3 fix
list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES" )

# Setup of the dependencies
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

SET(OSX FALSE)
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        SET(OSX TRUE)
        INCLUDE_DIRECTORIES("/usr/local/cuda/include")
        LINK_DIRECTORIES("/usr/local/cuda/lib/")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# EIGEN
FIND_PACKAGE(PkgConfig REQUIRED)
PKG_SEARCH_MODULE(EIGEN REQUIRED eigen3)
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIRS})
LINK_DIRECTORIES(${EIGEN_LIBRARY_DIRS})

# Protobuf
IF (OSX)
SET(PROTOBUF_IMPORT_DIRS /opt)
INCLUDE(${CMAKE_CURRENT_LIST_DIR}/FindProtobuf.cmake)
FIND_PACKAGE(Protobuf 3.1 REQUIRED)
ELSE(OSX)
# It does not want to be found, so we hardcode it...
SET(PROTOBUF_INCLUDE_DIRS /opt/include)
SET(PROTOBUF_LIBRARIES optimized;/opt/lib/libprotobuf.so;debug;/opt/lib/libprotobuf.so;-lpthread)
SET(PROTOBUF_PROTOC_LIBRARIES optimized;/opt/lib/libprotoc.so;debug;/opt/lib/libprotoc.so )
SET(PROTOBUF_LITE_LIBRARIES optimized;/opt/lib/libprotobuf-lite.so;debug;/opt/lib/libprotobuf-lite.so)
ENDIF(OSX)

MESSAGE(STATUS "Protobuf found ${PROTOBUF_INCLUDE_DIRS}")
MESSAGE(STATUS "Protobuf found ${PROTOBUF_LIBRARIES}")
MESSAGE(STATUS "Protobuf found ${PROTOBUF_PROTOC_LIBRARIES}")
MESSAGE(STATUS "Protobuf found ${PROTOBUF_LITE_LIBRARIES}")
INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIRS})

# OpenCV 3
FIND_PACKAGE(OpenCV 3 REQUIRED)
INCLUDE_DIRECTORIES( ${OpenCV3_INCLUDE_DIRS} )

# CUDA
find_package(CUDA REQUIRED)
IF(CUDA_FOUND)
MESSAGE(STATUS "Found CUDA")
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} /usr/local/cuda/lib64/libcudart.so)
LINK_DIRECTORIES(${CUDA_LIBRARIES})
MESSAGE(STATUS "Cuda found ${CUDA_INCLUDE_DIRS}, ${CUDA_LIBRARIES} ")
ENDIF(CUDA FOUND)

# Tensorflow
FIND_PACKAGE(Tensorflow REQUIRED)
INCLUDE_DIRECTORIES(${TensorFlow_INCLUDE_DIRS})
LINK_DIRECTORIES(${TensorFlow_LIBRARIES})


# Set output folders where to put the executables and libs
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Set C++ compile flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -march=native -O3 -Wno-deprecated-declarations --coverage")

#Libraries
SET(LIBS_LINKED_LIBS
${OpenCV_LIBRARIES}
${TensorFlow_LIBRARIES}
${PROTOBUF_LIBRARIES}
${CUDA_LIBRARIES}
)

#main
ADD_EXECUTABLE(cnn cnn.cpp)
TARGET_LINK_LIBRARIES(cnn ${LIBS_LINKED_LIBS} cuda)

# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.18)

# Set the project name
project(chat)

set(KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../kernels)

include_directories(${KERNEL_DIR} 
                    include
                    include/nn_modules
                    json/single_include
                    half-2.2.0/include
                    )

# Set the C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Ofast")

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    include_directories(include/ops/cuda include)
    aux_source_directory(${KERNEL_DIR}/cuda CUDA_SOURCES_FILES)
    aux_source_directory(src/nn_modules/cuda CUDA_SOURCES_FILES)
    aux_source_directory(src/ops/cuda CUDA_SOURCES_FILES)
    message("Cuda files :\n${CUDA_SOURCES_FILES}")
else()
  message(STATUS "No CUDA support")
endif()

add_library(CUDA_Shared_lib SHARED ${CUDA_SOURCES_FILES})

aux_source_directory(src COMMON_FILES)
add_library(Common_Shared_lib SHARED ${COMMON_FILES})
message("Common files :\n${COMMON_FILES}")
# aux_source_directory(. SRC)
# enable_language(CXX)
add_executable(${PROJECT_NAME}  application/chat.cc)

target_link_libraries(${PROJECT_NAME}  CUDA_Shared_lib Common_Shared_lib)
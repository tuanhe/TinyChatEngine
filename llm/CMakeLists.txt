# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.18)

option(ENABLE_PROFILE "Enable PROFILE" ON)

include(CheckLanguage)
# Set the project name
project(chat LANGUAGES CXX)

set(KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../kernels)

include_directories(${KERNEL_DIR} 
                    include
                    include/nn_modules
                    json/single_include
                    half-2.2.0/include
                    )

# Set the C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Ofast")


# find_package(CUDAToolkit REQUIRED)
# if (CUDAToolkit_FOUND)
# IF (CUDA_FOUND)
#     message(STATUS "    config: ${CUDA_DIR}")
#     message(STATUS "    version: ${CUDA_VERSION}")
#     message(STATUS "    libraries: ${CUDA_LIBRARIES}")
#     message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
#     #  Specify the arch of your CUDA hardware here
#     # SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++14 -gencode arch=compute_75,code=sm_75)
# ENDIF ()

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    FIND_PACKAGE(CUDA REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    enable_language(CUDA)
    add_definitions(-DQM_CUDA -DENABLE_BF16  -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_BFLOAT16_OPERATORS__  
                    -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -U__CUDA_NO_BFLOAT162_OPERATORS__ -U__CUDA_NO_BFLOAT162_CONVERSIONS__ )
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math")
    include_directories(include/ops/cuda include ${CUDA_INCLUDE_DIRS})
    aux_source_directory(${KERNEL_DIR}/cuda CUDA_SOURCES_FILES)
    aux_source_directory(src/nn_modules/cuda CUDA_SOURCES_FILES)
    aux_source_directory(src/ops/cuda CUDA_SOURCES_FILES)
    message("Cuda files :\n${CUDA_SOURCES_FILES}")
else()
  message(STATUS "No CUDA support")
endif()

add_library(CUDA_Shared_lib SHARED ${CUDA_SOURCES_FILES})
set_target_properties(CUDA_Shared_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_compile_features(CUDA_Shared_lib PUBLIC cxx_std_14 )
# set_target_properties( CUDA_Shared_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_compile_options(CUDA_Shared_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=sm_75>)
# cuda_add_library(CUDA_Shared_lib STATIC "${CUDA_SOURCES_FILES}")

aux_source_directory(${KERNEL_DIR} COMMON_FILES)
aux_source_directory(src COMMON_FILES)
aux_source_directory(src/ops COMMON_FILES)
aux_source_directory(src/nn_modules COMMON_FILES)

add_library(Common_Shared_lib SHARED ${COMMON_FILES})
message("Common files :\n${COMMON_FILES}")

## application
add_executable(${PROJECT_NAME}  application/chat.cc)
target_link_libraries(${PROJECT_NAME}  CUDA_Shared_lib Common_Shared_lib)


## profile
if (ENABLE_PROFILE)
  add_subdirectory(tests)
endif()
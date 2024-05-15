


function(GenerateTest FileName)
    get_filename_component(EXECUTABLE_NAME ${FileName} NAME_WE)
    MESSAGE("Function ${FileName} ${EXECUTABLE_NAME}")
    add_executable(${EXECUTABLE_NAME} ${FileName})
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE #${PROJECT_NAME} 
                                                     Common_Shared_lib
                                                     CUDA_Shared_lib
                                                    )
    install(TARGETS ${EXECUTABLE_NAME} DESTINATION lib/${PROJECT_NAME})
endfunction()

function(GenerateCudaTest FileName)
    get_filename_component(EXECUTABLE_NAME ${FileName} NAME_WE)
    MESSAGE("Function ${FileName} ${EXECUTABLE_NAME}")
    add_executable(${EXECUTABLE_NAME} ${FileName})
    target_link_libraries(${EXECUTABLE_NAME} #${PROJECT_NAME} 
                                                     CUDA_Shared_lib
                                                     Common_Shared_lib
                                                    )
    # install(TARGETS ${EXECUTABLE_NAME} DESTINATION lib/${PROJECT_NAME})
endfunction()

if(CMAKE_CUDA_COMPILER)
    AUX_SOURCE_DIRECTORY(./cuda  test_cuda_source_files)
    message(STATUS "CUDA compiler: ${test_cuda_source_files}")
    # set_target_properties(kernels
                    #    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    AUX_SOURCE_DIRECTORY(./non_cuda  test_non_cuda_source_files)
    message(STATUS "No CUDA support, ${test_non_cuda_source_files}")
endif()


AUX_SOURCE_DIRECTORY(.  test_source_files)
list(APPEND test_sources ${test_source_files})
message("Test files are: ${test_sources}")

foreach(FILE_NAME ${test_sources})
    GenerateTest(${FILE_NAME})
endforeach()

foreach(FILE_NAME ${test_cuda_source_files})
    GenerateTest(${FILE_NAME})
endforeach()
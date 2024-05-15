
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

AUX_SOURCE_DIRECTORY(.  test_source_files)
list(APPEND test_sources ${test_source_files})
message("Test files are: ${test_sources}")

foreach(FILE_NAME ${test_sources})
    GenerateTest(${FILE_NAME})
endforeach()
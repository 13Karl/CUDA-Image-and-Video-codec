cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(picsong LANGUAGES CXX CUDA)

add_subdirectory(ThirdParty/cub EXCLUDE_FROM_ALL)

find_path(NvToolExt_INCLUDE_DIR nvToolsExt.h HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(${PROJECT_NAME}_library STATIC
        src/Image/Image.cpp
        src/Image/Image.hpp
        src/IO/CommandLineParser.hpp
        src/IO/CommandLineParser.cpp
        src/IO/IOManager.hpp
        src/IO/IOManager.ipp
        src/SupportFunctions/AuxiliarFunctions.hpp
        src/SupportFunctions/AuxiliarFunctions.cpp)

set_target_properties(${PROJECT_NAME}_library PROPERTIES
	CXX_STANDARD 14)

target_include_directories(${PROJECT_NAME}_library PUBLIC ${NvToolExt_INCLUDE_DIR})

add_executable(${PROJECT_NAME}
	src/Launcher.cu
	src/BitStreamBuilder/BitStreamBuilder.cpp
	src/BitStreamBuilder/BitStreamBuilder.cu
	src/BitStreamBuilder/BitStreamBuilder.cuh
	src/BitStreamBuilder/BitStreamBuilder.hpp
	src/BPC/BPCEngine.cuh
	src/BPC/BPCEngine.hpp
	src/BPC/BPCEngine.ipp
	src/DWT/DWTGenerator.cuh
	src/DWT/DWTGenerator.hpp
	src/DWT/DWTGenerator.ipp
	src/Engines/CodingEngine.cu
	src/Engines/CodingEngine.cuh
	src/Engines/DecodingEngine.cu
	src/Engines/DecodingEngine.cuh
	src/Engines/Engine.cu
	src/Engines/Engine.cuh)

set_target_properties(${PROJECT_NAME} PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_STANDARD 14)

target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}_library CUB::CUB nvToolsExt)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)


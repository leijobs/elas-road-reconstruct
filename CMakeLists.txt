cmake_minimum_required(VERSION 2.8.3)
project(elas_reconstruction)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost 1.54.0 COMPONENTS system filesystem program_options REQUIRED)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
set(CMAKE_CXX_STANDARD 14)

set(VTK_DIR /home/hoho/Dependencies/VTK-8.2.0/build) # VTK 的路径

set(INC_DIR ./include)
set(LIBELAS_SRC_DIR src/elas)
set(CMAKE_CXX_FLAGS "-msse3")
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

FILE(GLOB LIBELAS_SRC_FILES "src/elas/*.cpp")

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${LIBELAS_SRC_DIR}
  ${Boost_INCLUDE_DIRS}
  ${INC_DIR}
)

add_executable(elas_reconstruction
  src/elas_reconstruction.cpp
  ${LIBELAS_SRC_FILES}
)

target_link_libraries(elas_reconstruction ${OpenCV_LIBS} ${PCL_LIBRARIES} pthread)

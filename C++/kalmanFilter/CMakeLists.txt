cmake_minimum_required (VERSION 3.0)
project (kalmanFilter)
 
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 
add_executable (kalmanFilter kalmanFilter.cpp)
target_link_libraries (kalmanFilter Eigen3::Eigen)
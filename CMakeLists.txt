cmake_minimum_required(VERSION 3.8)
project(project_mtc_pickplace_pointcloud)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Define required packages (alphabetical order)
set(REQUIRED_PACKAGES
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  moveit_task_constructor_core
  moveit_task_constructor_msgs
  mycobot_interfaces
  PCL
  pcl_conversions
  pcl_ros
  rclcpp
  sensor_msgs
  shape_msgs
  tf2_eigen
  tf2_geometry_msgs
)

# Find required packages
foreach(package ${REQUIRED_PACKAGES})
  find_package(${package} REQUIRED)
endforeach()

find_package(PCL REQUIRED COMPONENTS common features filters io kdtree search segmentation)

# Include directories
include_directories(include ${PCL_INCLUDE_DIRS})

# Add PCL definitions
add_definitions(${PCL_DEFINITIONS})
add_definitions(-DPCL_NO_PRECOMPILE)

# Create a shared library for common functionality
add_library(${PROJECT_NAME}_lib SHARED
  src/cluster_extraction.cpp
  src/get_planning_scene_client.cpp
  src/normals_curvature_and_rsd_estimation.cpp
  src/object_segmentation.cpp
  src/plane_segmentation.cpp
)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}_lib ${PCL_LIBRARIES})
ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  pcl_conversions
  pcl_ros
  sensor_msgs
  shape_msgs
  moveit_msgs
  mycobot_interfaces
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
)

# Executable: get_planning_scene_server
add_executable(get_planning_scene_server src/get_planning_scene_server.cpp)
target_link_libraries(get_planning_scene_server ${PROJECT_NAME}_lib)
ament_target_dependencies(get_planning_scene_server
  rclcpp
  pcl_conversions
  pcl_ros
  sensor_msgs
  shape_msgs
  moveit_msgs
  mycobot_interfaces
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
)

# Executable: mtc_node
add_executable(mtc_node src/mtc_node.cpp)
target_link_libraries(mtc_node ${PROJECT_NAME}_lib)
ament_target_dependencies(mtc_node
  rclcpp
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  moveit_task_constructor_core
  moveit_task_constructor_msgs
  mycobot_interfaces
  sensor_msgs
  tf2_eigen
  tf2_geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS ${PROJECT_NAME}_lib get_planning_scene_server mtc_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()

cmake_minimum_required(VERSION 3.8)
project(bt_main)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(behaviortree_ros2 REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(opennav_docking_msgs REQUIRED)

# Include directories
include_directories(
  include
)

add_executable(bt_main src/bt_main.cpp)
target_link_libraries(bt_main
  model_client
  bt_action_nav
  bt_action_firmware
  bt_receiver
  ${ARMADILLO_LIBRARIES})
ament_target_dependencies(bt_main
  ${THIS_PACKAGE_DEPS}
  "behaviortree_ros2"
  "behaviortree_cpp"
  "rclcpp"
  "std_msgs"
)
install(TARGETS
  bt_main
  DESTINATION lib/${PROJECT_NAME}
)

add_library(model_client src/model_client.cpp)
target_link_libraries(model_client
  ${ARMADILLO_LIBRARIES})
ament_target_dependencies(model_client
  "behaviortree_ros2"
  "behaviortree_cpp"
  "std_msgs"
  ${THIS_PACKAGE_DEPS}
)

add_library(bt_action_nav src/bt_action_nav.cpp)
target_link_libraries(bt_action_nav
  ${ARMADILLO_LIBRARIES})
ament_target_dependencies(bt_action_nav
  "behaviortree_ros2"
  "behaviortree_cpp"
  "rclcpp"
  "geometry_msgs"
  "tf2"
  "tf2_geometry_msgs"
  "opennav_docking_msgs"
  ${THIS_PACKAGE_DEPS}
)

add_library(bt_action_firmware src/bt_action_firmware.cpp)
target_link_libraries(bt_action_firmware
  ${ARMADILLO_LIBRARIES})
ament_target_dependencies(bt_action_firmware
  "behaviortree_ros2"
  "behaviortree_cpp"
  "rclcpp"
  "std_msgs"
  "geometry_msgs"
  "tf2"
  ${THIS_PACKAGE_DEPS}
)

add_library(bt_receiver src/bt_receiver.cpp)
target_link_libraries(bt_receiver
  ${ARMADILLO_LIBRARIES})
ament_target_dependencies(bt_receiver
  "behaviortree_ros2"
  "behaviortree_cpp"
  "rclcpp"
  "geometry_msgs"
  "tf2"
  "tf2_ros"
  "tf2_geometry_msgs"
  ${THIS_PACKAGE_DEPS}
)

install(DIRECTORY
  bt_config
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(behaviortree_ros2)
ament_export_dependencies(behaviortree_cpp)

ament_package()

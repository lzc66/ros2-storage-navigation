cmake_minimum_required(VERSION 3.8)
project(project)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Custom interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SpawnItem.srv"
  "action/FetchTask.action"
)

add_executable(main_node src/main_node.cpp)
target_include_directories(main_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(main_node PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  main_node
  "rclcpp"
  "std_msgs"
  "geometry_msgs"
)

# Install Python scripts
install(PROGRAMS
  scripts/item_spawner.py
  scripts/auto_tune_nav2.py
  scripts/generate_map.py
  scripts/vision_node.py
  scripts/brain_node.py
  scripts/dynamic_obstacle.py
  scripts/sim_manager.py
  scripts/calibration_check.py
  scripts/task_emitter.py
  scripts/stress_test_preemption.py
  scripts/e2e_pick_place_test.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install directories
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY
  worlds
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY
  models
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY
  maps
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY
  params
  DESTINATION share/${PROJECT_NAME}
)

# TensorRT engine (YOLOv8n FP16)
install(FILES
  scripts/yolov8n.engine
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS main_node
  DESTINATION lib/${PROJECT_NAME})

rosidl_target_interfaces(main_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

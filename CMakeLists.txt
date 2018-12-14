cmake_minimum_required(VERSION 2.8.3)
project(ds_object_detection)

## Add support for C++11, supported in ROS Kinetic and newer
# add_definitions(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  message_generation
  sensor_msgs
  std_msgs
)



add_message_files(
  DIRECTORY
    msg
  FILES 
    PointCloud2Array.msg
    ColorDepthImage.msg
    ColorDepthImageArray.msg
    Detection.msg
    DetectionArray.msg
    Float64ArrayStamped.msg
    PositionMeasurement.msg 
    PositionMeasurementArray.msg 
    PersonStamped.msg
    Person.msg
    People.msg
    Mask.msg
    Rect.msg
    Skeleton.msg
)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    sensor_msgs
    std_msgs
)



catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    message_runtime
    sensor_msgs
    std_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)





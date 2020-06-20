cmake_minimum_required(VERSION 3.0.2)
project(demo_lidar)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
  roscpp
        tf
)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(Cholmod REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES demo_lidar
#  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

include_directories(
 include
        /usr/include/suitesparse/
  ${catkin_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${cv_bridge_INCLUDE_DIRS}
        ${SuiteSparse_INCLUDE_DIRS}
        ${Cholmod_INCLUDES}
)
add_library(libisam
        isamlib/Anchor.cpp
        isamlib/Cholesky.cpp
        isamlib/ChowLiuTree.cpp
        isamlib/covariance.cpp
        isamlib/Covariances.cpp
        isamlib/glc.cpp
        isamlib/GLCReparam.cpp
        isamlib/Node.cpp
        isamlib/numericalDiff.cpp
        isamlib/Optimizer.cpp
        isamlib/OrderedSparseMatrix.cpp
        isamlib/Slam.cpp
        isamlib/SparseMatrix.cpp
        isamlib/SparseSystem.cpp
        isamlib/SparseVector.cpp
        isamlib/util.cpp
        )

target_link_libraries(libisam
        ${CHOLMOD_LIBRARIES})

add_executable(featureTracking src/featureTracking.cpp)
target_link_libraries(featureTracking 
#/usr/local/lib/libisam.a
        libisam
${PCL_LIBRARIES}
${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        ${cv_bridge_LIBRARIES}
        ${SuiteSparse_LIBRARIES}
)

add_executable(visualOdometry src/visualOdometry.cpp)
target_link_libraries(visualOdometry
        #/usr/local/lib/libisam.a
        libisam
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        ${cv_bridge_LIBRARIES}
        ${SuiteSparse_LIBRARIES}
        )

add_executable(bundleAdjust src/bundleAdjust.cpp)
target_link_libraries(bundleAdjust
        #/usr/local/lib/libisam.a
        libisam
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        ${cv_bridge_LIBRARIES}
        ${SuiteSparse_LIBRARIES}
        )

add_executable(processDepthmap src/processDepthmap.cpp)
target_link_libraries(processDepthmap
        #/usr/local/lib/libisam.a
        libisam
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        ${cv_bridge_LIBRARIES}
        ${SuiteSparse_LIBRARIES}
        )

add_executable(stackDepthPoint src/stackDepthPoint.cpp)
target_link_libraries(stackDepthPoint
        #/usr/local/lib/libisam.a
        libisam
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        ${cv_bridge_LIBRARIES}
        ${SuiteSparse_LIBRARIES}
        )

add_executable(transformMaintenance src/transformMaintenance.cpp)
target_link_libraries(transformMaintenance
        #/usr/local/lib/libisam.a
        libisam
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        ${cv_bridge_LIBRARIES}
        ${SuiteSparse_LIBRARIES}
        )

add_executable(registerPointCloud src/registerPointCloud.cpp)
target_link_libraries(registerPointCloud
        #/usr/local/lib/libisam.a
        libisam
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
        ${cv_bridge_LIBRARIES}
        ${SuiteSparse_LIBRARIES}
        )

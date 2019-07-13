cmake_minimum_required(VERSION 3.0)

set(MAJOR_VERSION 0)
set(MINOR_VERSION 0)
set(PATCH_VERSION 1)

project(OpenVO
	VERSION ${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}
	DESCRIPTION "Visual Odometry based on OpenCV."
	LANGUAGES CXX)

add_subdirectory(src)


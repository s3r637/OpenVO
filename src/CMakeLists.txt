
add_library(openvo STATIC
	openvo/Tracker.cpp
	openvo/Tracker.hpp
)
target_compile_features(openvo PUBLIC cxx_std_17)
set_target_properties(openvo
	PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(ovo main.cxx)
target_compile_features(ovo PUBLIC cxx_std_17)
set_target_properties(ovo
	PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
	LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
target_link_libraries(ovo PUBLIC
	Boost::filesystem
	Boost::program_options
	Eigen3::Eigen
	${OpenCV_LIBS}
	openvo
)

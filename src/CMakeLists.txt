
add_library(openvo STATIC
	openvo/Tracker.cpp
	openvo/Tracker.hpp
)
target_compile_features(openvo PUBLIC cxx_std_11)

add_executable(vo main.cxx)
target_link_libraries(vo PUBLIC openvo)

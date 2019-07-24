.PHONY: release debug reldeb doc clean

release:
	(mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j)

debug:
	(mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j)

reldeb:
	(mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .. && make -j)

doc:
	doxygen

clean:
	(rm -rf build doc)
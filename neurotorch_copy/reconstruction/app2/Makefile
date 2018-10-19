all:
	clang++ -g -O3 -Wall -shared -std=c++11 -fPIC  \
		-I./basic_c_fun \
		-ltiff \
		`python -m pybind11 --includes` \
		./basic_c_fun/stackutil.cpp \
		./basic_c_fun/mg_utilities.cpp \
		./basic_c_fun/mg_image_lib.cpp \
		*.cpp -o app2`python3-config --extension-suffix`

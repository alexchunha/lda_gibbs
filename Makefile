all:
	g++ -O3 -Wall -shared -std=c++11 -fPIC `python3.6m -m pybind11 --includes` src/Lda.cpp -o Lda`python3.6m-config --extension-suffix` -undefined dynamic_lookup

OFLG=-g
CXXFLG=${OFLG}
test_avx512_ip:avx512_inner_product.cpp
	g++ ${CXXFLG} -fopenmp -mavx512f $^ -o $@
clean:
	-rm test_avx512_ip
run:demo
	./test_avx512_ip

OFLG=-g
CFLG=${OFLG}
test_amx_ip:amx_inner_product.c
	gcc ${CFLG} -mamx-tile -mamx-int8 $^ -o $@
clean:
	-rm test_amx_ip
run:demo
	./test_amx_ip

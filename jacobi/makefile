CAP=13
OPT=-O

all: j1 j2 j3 j4 j5 jf
	echo "make j1,j2,j3,j4,j5,jf"

clean:
	rm -f *.o j1 j2 j3 j4 j5 jf

j1:	jhost.o jkernel.o jdriver.o
	nvcc $(OPT) -o $@ $^

j2:	jhost.o jkernel.o jdrivert.o
	nvcc $(OPT) -o $@ $^

j3:	jhost.o jkernelt.o jdrivert.o
	nvcc $(OPT) -o $@ $^

j4:	jhost.o jkernelt2.o jdrivert.o
	nvcc $(OPT) -o $@ $^

j5:	jhost.o jkernelt3.o jdrivert.o
	nvcc $(OPT) -o $@ $^

jf:	jhost.o jkernelf.o jdrivert.o
	nvcc $(OPT) -o $@ $^


%.o: %.c
	gcc $(OPT) -std=c99 -c $<

%.o: %.cu
	nvcc $(OPT) -c -Xptxas=-v -arch=sm_$(CAP) $<




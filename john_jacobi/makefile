CAP=13
OPT=-O
M=18
N=18
T=14

all: jacobi

clean:
	rm -f *.o *.log logs/*.log

jacobi:	jacobi.cu
	nvcc $(OPT) jacobi.cu -o jacobi.o

%.o: %.c
	gcc $(OPT) -std=c99 -c $<

%.o: %.cu
	nvcc $(OPT) -c -Xptxas=-v -arch=sm_$(CAP) $<

run:
	mkdir -p logs
	./jacobi.o ${M} ${N} ${T} 2>&1 | tee logs/jacobi_${M}_${N}_${T}.log

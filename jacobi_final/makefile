OPT=-O

M=1024
N=1024

#M=2048
#N=2048

#M=4096
#N=4096

all: seq original5 original6 1k_jacobi5 1k_jacobi6

seq: seq_jacobi.c
	gcc ${OPT} -std=c99 seq_jacobi.c -o seq_jacobi.o -lm

original5: original_jacobi5.cu
	nvcc original_jacobi5.cu -o original_jacobi5.o

original6: original_jacobi6.cu
	nvcc original_jacobi6.cu -o original_jacobi6.o

1k_jacobi5: 1k_jacobi5.cu
	nvcc 1k_jacobi5.cu -o 1k_jacobi5.o

1k_jacobi6: 1k_jacobi6.cu
	nvcc 1k_jacobi6.cu -o 1k_jacobi6.o

run: seq_jacobi.o original_jacobi5.o original_jacobi6.o 1k_jacobi5.o 1k_jacobi6.o
	mkdir -p logs
	./seq_jacobi.o       ${M} ${N} 2>&1 | tee logs/seq_jacobi_${M}_${N}.log
	./original_jacobi5.o ${M} ${N} 2>&1 | tee logs/original_jacobi5_${M}_${N}.log
	./original_jacobi6.o ${M} ${N} 2>&1 | tee logs/original_jacobi6_${M}_${N}.log
	./1k_jacobi5.o       ${M} ${N} 2>&1 | tee logs/1k_jacobi5_${M}_${N}.log
	./1k_jacobi6.o       ${M} ${N} 2>&1 | tee logs/1k_jacobi6_${M}_${N}.log

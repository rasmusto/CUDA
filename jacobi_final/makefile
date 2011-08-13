OPT=-O

#M=18
#N=18

M=1026
N=1026

#M=2050
#N=2050

#M=4098
#N=4098

all: seq original5 original6 1k_jacobi5 1k_jacobi6

seq: seq_jacobi.c
	gcc ${OPT} -std=c99 seq_jacobi.c -o seq_jacobi.o -lm

original5: original_jacobi5.cu
	nvcc ${OPT} original_jacobi5.cu -o original_jacobi5.o

original6: original_jacobi6.cu
	nvcc ${OPT} original_jacobi6.cu -o original_jacobi6.o

1k_jacobi5: 1k_jacobi5.cu
	nvcc ${OPT} 1k_jacobi5.cu -o 1k_jacobi5.o

1k_jacobi6: 1k_jacobi6.cu
	nvcc ${OPT} 1k_jacobi6.cu -o 1k_jacobi6.o

%.o: %.cu
	nvcc $(OPT) -c -Xptxas=-v -arch=sm_$(CAP) $<

run: seq_jacobi.o original_jacobi5.o original_jacobi6.o 1k_jacobi5.o 1k_jacobi6.o
	mkdir -p logs
	#./seq_jacobi.o       ${M} ${N} 2>&1 | tee logs/seq_jacobi_${M}_${N}.log
	./original_jacobi5.o ${M} ${N} 2>&1 | tee logs/original_jacobi5_${M}_${N}.log
	./original_jacobi6.o ${M} ${N} 2>&1 | tee logs/original_jacobi6_${M}_${N}.log
	./1k_jacobi5.o       ${M} ${N} 2>&1 | tee logs/1k_jacobi5_${M}_${N}.log
	./1k_jacobi6.o       ${M} ${N} 2>&1 | tee logs/1k_jacobi6_${M}_${N}.log

clean:
	rm -f *.o

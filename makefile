XDIMM=18
YDIMM=18
default: JacobiRel.cu Sj3.cu Sj5.cu Sj6.cu
#	nvcc JacobiRel.cu -o JacobiRel
#	nvcc Sj3.cu -o Sj3
#	nvcc Sj5.cu -o Sj5
#	nvcc Sj6.cu -o Sj6
	nvcc Jj6.cu -o Jj6
run: run_JacobiRel 
run_JacobiRel: JacobiRel
	./JacobiRel
clean:
	rm -f JacobiRel Sj3 Sj5 Sj6

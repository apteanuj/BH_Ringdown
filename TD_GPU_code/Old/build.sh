module unload intel/composer_xe_2013_sp1
module unload openmpi/2.0.1-intel
module unload cuda/7.5.18
module load cuda/9.0.176

echo compiling..
g++ -O -g -c main.c proj.c
ar r lib.a main.o proj.o
rm main.o proj.o

nvcc -g -O -c body.cu -arch=sm_70 

ar r lib.a body.o
rm body.o

nvcc -O -c -w kern.cu -arch=sm_70 

ar r lib.a kern.o
rm kern.o

echo linking...
g++ -g -O -o a0.9_thi060_thf150.3_n\
	lib.a\
	-L/share/pkg/cuda/9.0.176/lib64 -lcudart

echo cleaning.. 
rm lib.a

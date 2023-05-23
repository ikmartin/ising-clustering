CXX ?= g++
CC ?= gcc
CFLAGS = -w -fopenmp -O3 -pie -fPIE
LIBS = blas/blas.a

all: clean LPsparse

LPsparse: CG.o blas/blas.a
	$(CXX) $(CFLAGS) -o LPsparse LPsparse.cpp CG.o $(LIBS)

CG.o: $(LIBS)
	        $(CXX) $(CFLAGS) -c -o CG.o CG.cpp $(LIBS)

blas/blas.a: blas/*.c blas/*.h
	        make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';
test:
	g++ $(CFLAGS) -o test test.cpp
clean:
		rm -f *~ *.o LPsparse blas/*.a

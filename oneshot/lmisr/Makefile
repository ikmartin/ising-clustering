CC = gcc

CFLAGS = -O3 -std=c99 -fPIC

LFLAGS =  -lm -lopenblas 

SRCS = IsingLPSolve.c util.c
SOBJ = isingLPA.so

OBJS = $(SRCS:.c=.o)

INC  = -I/opt/OpenBLAS/include
LIBS = -L/opt/OpenBLAS/lib

ifeq ($(HOSTNAME),nemo.lps.umd.edu)
INC  += -I/opt/OpenBLAS/include
LIBS += -L/opt/OpenBLAS/lib
endif

MAIN = isingLPA

.PHONY: depend clean

all:    $(MAIN) $(SOBJ)

$(MAIN): $(OBJS) 
	$(CC) $(CFLAGS) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

$(SOBJ): $(OBJS)
	$(CC) $(CFLAGS) $(OPT) -shared $(OBJS) $(INC) -o $@ $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) -c $< $(INC) -o $@

clean:
	$(RM) *.o *~ $(MAIN) $(SOBJ)

depend: $(SRCS)
	makedepend $^

CC = gcc

CFLAGS = -Ofast -std=c99 -fPIC  -g -pg

LFLAGS =  -lm -lopenblas 

SRCS = solver.c
SOBJ = solver.so

OBJS = $(SRCS:.c=.o)

INC  = -I/opt/OpenBLAS/include
LIBS = -L/opt/OpenBLAS/lib

ifeq ($(HOSTNAME),nemo.lps.umd.edu)
INC  += -I/opt/OpenBLAS/include
LIBS += -L/opt/OpenBLAS/lib
endif

MAIN = solver

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

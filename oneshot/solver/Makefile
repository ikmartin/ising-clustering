CC = gcc

CFLAGS = -Ofast -std=gnu99 -fPIC # -g -pg

LFLAGS =  -lm -lopenblas -lpthread

SRCS = solver.c
SOBJ = solver.so

OBJS = $(SRCS:.c=.o)

INC  = -I/opt/OpenBLAS/include
LIBS = -L/opt/OpenBLAS/lib 

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

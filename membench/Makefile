# Note: You will need to use a compiler with OpenMP in order to
# get the omp_get_wtime() routines to work.

CC = gcc
CFLAGS = -std=c99 -O2 -fopenmp
LIBS = -lrt

.PHONY: run clean clean-all

membench: membench.c
	$(CC) $(CFLAGS) -o membench membench.c 

membench.pdf: membench.py timings.csv
	./membench.py

run: membench
	csub ./membench timings.csv

clean:
	rm -f membench membench.eps membench.pdf
	rm -f csub-*

clean-all: clean
	rm -f timings.csv

# ---
# Platform-dependent configuration
#
# If you have multiple platform-dependent configuration options that you want
# to play with, you can put them in an appropriately-named Makefile.in.
# For example, the default setup has a Makefile.in.icc and Makefile.in.gcc.

PLATFORM=gcc
include Makefile.in.$(PLATFORM)

.PHONY:	all
all:	matmul-mine matmul-basic matmul-blocked matmul-blas matmul-f2c

# ---
# Rules to build the drivers

matmul-%: $(OBJS) dgemm_%.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS)

matmul-f2c: $(OBJS) dgemm_f2c.o dgemm_f2c_desc.o fdgemm.o
	$(FC) -o $@ $^ $(LDFLAGS) $(LIBS) 

matmul-blas: $(OBJS) dgemm_blas.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS) $(LIBBLAS)

# --
# Rules to build object files

%.o: %.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $<

%.o: %.f
	$(FC) -c $(FFLAGS) $<

dgemm_blas.o: dgemm_blas.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCBLAS) $< 

# ---
# Rules for building timing CSV outputs

timing-%.csv: matmul-%
	csub ./runner.sh ./$< $*

# ---

.PHONY:	clean realclean 
clean:
	rm -f matmul-* *.o
	rm -f csub-*

realclean:	clean
	rm -f *~ timing-*.csv info-*.out timing.pdf


ifndef PLATFORM
include Makefile.in
else
include Makefile.in.$(PLATFORM)
endif

.PHONY:	all
all:	matmul matmul-basic matmul-blocked matmul-blas matmul-f2c

# ---
# Rules to build the drivers

matmul: $(OBJS) dgemm.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS)

matmul-basic: $(OBJS) basic_dgemm.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS)

matmul-blocked:	$(OBJS) blocked_dgemm.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS)

matmul-blas:	$(OBJS) blas_dgemm.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS) $(LIBBLAS)

matmul-f2c:	$(OBJS) f2c_dgemm.o fdgemm.o
	$(FC) -o $@ $^ $(LDFLAGS) $(LIBS) 

# --
# Rules to build object files

%.o:%.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $<

%.o:%.f
	$(FC) -c $(FFLAGS) $<

blas_dgemm.o: blas_dgemm.c
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCBLAS) $< 

# ---
# This is just a suggestion on how to generate timing plots...  Feel
# free to improve on these, so long as you show MFlop/s v. matrix size.
# Submit matmul via qsub to generate the raw timing data.

.PHONY: run
run: matmul
	csub ./runner.sh ./matmul mine

.PHONY: run-basic
run-basic: matmul-basic
	csub ./runner.sh ./matmul-basic basic

.PHONY: run-blocked
run-blocked: matmul-blocked
	csub ./runner.sh ./matmul-blocked blocked

.PHONY: run-blas
run-blas: matmul-blas
	csub ./runner.sh ./matmul-blas openblas

.PHONY: run-f2c
run-f2c: matmul-f2c
	csub ./runner.sh ./matmul-f2c f2c

# --
# Rules to compile timepgf (if everything is available)
timepgf.pdf: timepgf.tex
	pdflatex timepgf.tex
	pdflatex timepgf.tex

# ---

.PHONY:	clean realclean tgz
clean:
	rm -f matmul matmul-basic matmul-blocked matmul-blas \
		matmul-f2c *.o
	rm -f csub-*
	rm -f timepgf.aux timepgf.log timepgf.out

realclean:	clean
	rm -f *~ timing-*.csv info-*.out

tgz: realclean
	(cd ..; tar -czf matmul.tgz matmul)

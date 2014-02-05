SLIDES = lec01.pdf lec02.pdf lec03.pdf lec04.pdf lec05.pdf

export TEXINPUTS := .:figs:$(TEXINPUTS):

all: ${SLIDES}
	cp ${SLIDES} web

lec01.pdf: lec01.tex lec01plot.dat

lec01plot.dat: lec01plot.py
	python lec01plot.py

lec03.pdf: lec03.tex lec03membench.pdf lec03membench2.pdf

lec03membench.pdf: lec03membench.py lec03membench.csv
	python lec03membench.py lec03membench

lec03membench2.pdf: lec03membench2.py lec03membench.csv
	python lec03membench2.py lec03membench

lec04.pdf: lec04.tex lec04membench.pdf lec04membench2.pdf \
	lec04matmul.pdf lec04matmul_gcc.pdf

lec04membench.pdf: lec03membench.py lec04membench.csv
	python lec03membench.py lec04membench

lec04membench2.pdf: lec04membench2.py lec04membench.csv
	python lec04membench2.py lec04membench gist_stern

lec04matmul.pdf: 
	(cd lec04timings; python plotter.py basic blocked f2c dsb blas)
	mv lec04timings/timing.pdf lec04matmul.pdf

lec04matmul_gcc.pdf: 
	(cd lec04timings_gcc; python plotter.py basic blocked f2c blas)
	mv lec04timings_gcc/timing.pdf lec04matmul_gcc.pdf

%.pdf: %.tex 
	pdflatex $<

clean:
	rm -f *~
	rm -f lec*.aux lec*.log lec*.nav lec*.out lec*.snm lec*.toc lec*.vrb
	rm -f lec01plot.dat lec01plot.pdf
	rm -f lec03membench.pdf
	rm -f lec04membench2*.pdf lec04matmul.pdf

realclean: clean
	rm -f lec*.pdf

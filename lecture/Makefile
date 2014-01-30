SLIDES = lec01.pdf lec02.pdf lec03.pdf

export TEXINPUTS := .:figs:$(TEXINPUTS):

all: ${SLIDES}
	cp ${SLIDES} web

lec01.pdf: lec01.tex lec01plot.dat

lec01plot.dat: lec01plot.py
	python lec01plot.py

lec03.pdf: lec03.tex lec03membench.pdf

lec03membench.pdf: lec03membench.py lec03membench.csv
	python lec03membench.py lec03membench

%.pdf: %.tex 
	pdflatex $<

clean:
	rm -f *~
	rm -f lec*.aux lec*.log lec*.nav lec*.out lec*.snm lec*.toc lec*.vrb
	rm -f lec01plot.dat lec01plot.pdf
	rm -f lec03membench.pdf

realclean: clean
	rm -f lec*.pdf

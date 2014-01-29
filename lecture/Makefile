SLIDES = lec01.pdf lec02.pdf

export TEXINPUTS := .:figs:$(TEXINPUTS):

all: ${SLIDES}
	cp ${SLIDES} web

lec01.pdf: lec01.tex lec01plot.dat

lec01plot.dat: lec01plot.py
	python lec01plot.py

%.pdf: %.tex 
	pdflatex $<

clean:
	rm -f *~
	rm -f lec*.aux lec*.log lec*.nav lec*.out lec*.snm lec*.toc lec*.vrb
	rm -f lec01plot.dat lec01plot.pdf

realclean: clean
	rm -f lec*.pdf

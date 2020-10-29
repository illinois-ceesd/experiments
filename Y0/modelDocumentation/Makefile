all: all-redirect

BASENAME = modelDocument
LATEX = pdflatex
FIGURES := $(shell ls Figures/*.pdf Figures/*.png)
#SUBTALKS := $(shell ls sub-modelDoc*.tex)

.SUFFIXES: .pdf .tex

.tex.pdf: 
	$(LATEX) $*

all-redirect: $(BASENAME).pdf $(BASENAME)Slides.pdf

#$(BASENAME).pdf: $(BASENAME).tex $(SUBTALKS) $(FIGURES)
$(BASENAME).pdf: $(BASENAME).tex $(FIGURES)
#$(BASENAME)Slides.pdf: $(BASENAME)Slides.tex $(SUBTALKS) $(FIGURES)
$(BASENAME)Slides.pdf: $(BASENAME)Slides.tex $(FIGURES)

clean:
	rm -f *.vrb *.tex~ *.toc *.aux *.log *.nav *.out *.snm *.flc

realclean: clean
	rm -f $(BASENAME).pdf
	rm -f $(BASENAME)Slides.pdf


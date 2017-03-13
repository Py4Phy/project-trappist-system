NAME  := project_2
LATEX := $(NAME).tex
PDF   := $(NAME).pdf


%.pdf: %.tex
	pdflatex -shell-escape $<
	bibtex $*
	pdflatex -shell-escape $<
	pdflatex -shell-escape $<

.phony: all

all: $(PDF)

see: $(PDF)
	open $^



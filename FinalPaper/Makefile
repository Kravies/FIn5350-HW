all:
	Rscript -e 'rmarkdown::render("draft2.Rmd", output_format=c("html_document", "pdf_document", "word_document"))'

html:
	Rscript -e 'rmarkdown::render("draft2.Rmd", output_format="html_document")'

pdf:
	Rscript -e 'rmarkdown::render("draft2.Rmd", output_format="pdf_document")'

word:
	Rscript -e 'rmarkdown::render("draft2.Rmd", output_format="word_document")'

clean:
	rm draft2.html draft2.pdf draft2.docx draft2.tex

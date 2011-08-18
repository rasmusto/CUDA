DOCNAME=report
OUTLINE=skeleton

skeleton:
	xelatex -shell-escape ${OUTLINE}.tex

report:
	pdflatex -shell-escape ${DOCNAME}.tex

clean:
	rm -rf ${DOCNAME}.aux ${DOCNAME}.pdf ${DOCNAME}.log
	rm -rf ${OUTLINE}.aux ${OUTLINE}.pdf ${OUTLINE}.log

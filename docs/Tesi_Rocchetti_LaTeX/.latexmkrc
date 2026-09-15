# Automatically compile frontespizio title page documents (*-frn.tex)
add_cus_dep('tex', 'pdf', 0, 'frn2pdf');
sub frn2pdf {
    return system("pdflatex $_[0]");
}

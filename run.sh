#!/bin/bash

wget https://home.strw.leidenuniv.nl/~belcheva/galaxy_data.txt

python3 NURhandin4_1.py
python3 NURhandin4_2.py
python3 NURhandin4_3.py > NURhandin4problem3.txt

echo "Generating the pdf"

pdflatex handin4_JvV_s2272881.tex
pdflatex handin4_JvV_s2272881.tex



# langmuir-evaluation
Software to evaluate Langmuir measurement data of sputter targets

2017, Johannes Postler, Stefan Raggl, University of Innsbruck

**Usage**

usage: main.py [-h] [--filename FILENAME] [--plot PLOT] [--output OUTPUT]

Evaluation script for langmuir measurements

optional arguments:
  -h, --help            show this help message and exit
  --filename FILENAME, -f FILENAME
                        Specify a filename to be evaluated. Defaults to
                        testdata.txt
  --plot PLOT, -p PLOT  Specify an angle to be plotted in detail
  --output OUTPUT, -o OUTPUT
                        Specify a filename for the output data. Defaults to
                        results.txt
                        
**TODO**
* re-sort all ax_n variables in order of appearance or name appropriately
* use errors for weighted fit
* GUI
* error handling

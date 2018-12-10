# AYNEC
Tools from the AYNEC suite

This repository contains the DataGen and ResTest tools, which are implemented as python scripts. To run them, check the parameters at the start of the python file, and run it from console. The python files contains documentation about every parameter and function.

The following files with format examples are provided: "WN11.txt" and "mockup-results.txt", corresponding to the input of the DataGen and ResTest tools. In "WN11.txt", each line contains a triple in the following order: source, relation, target. In "mockup-results.txt", each line contains the source, relation, target, ground-truth (gt), and a column with the result of each compared technique. Please, note that the file is expected to have the same header, but with different techniques.

This software is licensed under the GPLv3 licence. It is presented in the article "AYNEC: All You Need for Evaluating Completion Techniques in Knowledge Graphs", sent for the ESWC19 conference and currently under revision.

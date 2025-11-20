import re

"""
    Extract the regexes from Snort.
"""

file = open("../data/snort3-community.rules", "r")
file_write = open("../data/filtered/snort.txt", "w")

regex_capture = re.compile(r"pcre:\".*\";")
quantifier_capture = re.compile(r"\{[0-9]*,[0-9]*\}")

for line in file:
    regex_line = regex_capture.findall(line)

    if regex_line:
        regex = regex_line[0]
        regex = regex[6:-2]

        quantifiers = quantifier_capture.findall(line)
        if quantifiers:
            file_write.write(regex + "\n")

file.close()
file_write.close()
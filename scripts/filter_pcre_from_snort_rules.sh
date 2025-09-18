#!/bin/bash
sed -nE 's/.+pcre:"\/([^"\n]+)\/\w+".+/\1/p' | sed 's/\\\//\//g' | sort | uniq

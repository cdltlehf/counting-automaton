for file in $(ls -Sr data/normalized/*.txt); do
  output=${file/normalized/analysis-pattern}
  output=${output%.txt}.csv
  mkdir -p $(dirname $output)
  echo $file $output
  python scripts/analysis_pattern.py < $file > $output
  echo
done

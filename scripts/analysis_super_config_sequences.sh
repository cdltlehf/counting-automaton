logfile=/dev/null

if [ "$#" -ge 1 ]; then
  logfile=$1
fi

for file in $(ls -Sr data/super_config_sequence/*); do
  echo $file
  python scripts/analysis_super_config_sequences.py < $file 2> $logfile
  echo
done

> test_results_xss.txt

# Read each line from runs.txt
while IFS= read -r run
do
  # Find all "test_results.csv" files in the directory and its subdirectories
  find "$run" -type f -name "test_results.csv" -printf "%p\n" >> test_results_xss.txt
done < runs_xss.txt
#!/bin/bash

# Check if a directory has been provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory> [filter]"
    exit 1
fi

# The directory to search for eval.json files
SEARCH_DIR=$1

# Optional filter for directory names
FILTER=$2

# Temporary file to hold individual JSON entries
TMP_FILE=$(mktemp)

# Start the JSON array
echo "[" > "$TMP_FILE"

# Build the find command with an optional filter
FIND_CMD="find \"$SEARCH_DIR\" -type f -name 'eval.json'"
if [ ! -z "$FILTER" ]; then
    FIND_CMD="find \"$SEARCH_DIR\" -type d -name \"$FILTER*\" -exec find {} -type f -name 'eval.json' \;"
fi

# Evaluate the find command and process the files
eval $FIND_CMD | while read -r file_path; do
    # Extract the setting name from the path
    setting_name=$(echo "$file_path" | sed -e 's|.*/\(.*\)/checkpoint.*|\1|')

    # Remove the word 'augmented' from the setting name
    setting_name=${setting_name//_augmented/}

    # Append a comma if it's not the first entry
    if [ $(wc -l < "$TMP_FILE") -gt 1 ]; then
        echo "," >> "$TMP_FILE"
    fi

    # Create a JSON entry with the path and contents of eval.json
    echo "{" >> "$TMP_FILE"
    echo "  \"Path\": \"$file_path\"," >> "$TMP_FILE"
    echo "  \"Setting\": \"$setting_name\"," >> "$TMP_FILE"
    content=$(jq '.' "$file_path")
    echo "  \"Content\": $content" >> "$TMP_FILE"
    echo "}" >> "$TMP_FILE"
done

# End the JSON array
echo "]" >> "$TMP_FILE"

# Format and write the final JSON to report.json
jq '.' "$TMP_FILE" > "report.json"

# Remove the temporary file
rm "$TMP_FILE"

echo "Report has been generated: report.json"

# Call the Python script to generate LaTeX table from report.json
python -m text2sparql.latex_table

# Check if the LaTeX table was successfully generated
if [ -f "report_table.tex" ]; then
    echo "LaTeX table has been generated: report_table.tex"
else
    echo "Failed to generate LaTeX table."
fi

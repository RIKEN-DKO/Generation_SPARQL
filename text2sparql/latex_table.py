import json

# Function to generate LaTeX table from the JSON data
def generate_latex_table(data):
    # Start of the LaTeX table
    latex_content = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Setting & BLEU & SP-BLEU & METEOR & ROUGE & F1-score \\\\ \\midrule"
    ]
    
    # Adding rows to the table from JSON data
    for entry in data:
        results = entry['Content']['results']
        setting = entry['Setting'].replace("_", "\\_")
        bleu = f"{results['BLEU']:.3f}"
        sp_bleu = f"{results['SP-BLEU']:.3f}"
        meteor = f"{results['METEOR']:.3f}"
        rouge = f"{results['ROUGE']:.3f}"
        f1_score = f"{results['f1']:.3f}"
        row = f"{setting} & {bleu} & {sp_bleu} & {meteor} & {rouge} & {f1_score} \\\\"
        latex_content.append(row)
    
    # End of the LaTeX table
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Evaluation metrics for different settings}",
        "\\label{tab:my_label}",
        "\\end{table}"
    ])
    
    # Join all parts into the final LaTeX table string
    return "\n".join(latex_content)

# Load JSON data from report.json
with open('report.json', 'r') as json_file:
    data = json.load(json_file)

# Generate LaTeX table and save to report_table.tex
latex_table = generate_latex_table(data)
with open('report_table.tex', 'w') as tex_file:
    tex_file.write(latex_table)

print("LaTeX table has been generated: report_table.tex")

#!/usr/bin/env python3
"""
Visualize hyperparameter search results as tables.

Creates formatted tables from JSON search results files.
"""

import json
import argparse
from pathlib import Path


def load_results(json_file):
    """Load search results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def create_dim_search_table(results, output_file=None):
    """Create a table for dimension search results."""
    if not results:
        print("No results to display")
        return
    
    # Extract data
    table_data = []
    for r in results:
        row = [
            r.get('d_model', 'N/A'),
            r.get('d_ff', 'N/A'),
            r.get('sentiment_hidden_dim', 'N/A'),
            r.get('fusion_hidden_dim', 'N/A'),
            '✅' if r.get('success', False) else '❌',
            r.get('run_name', 'N/A')
        ]
        table_data.append(row)
    
    headers = ['d_model', 'd_ff', 'sent_hidden', 'fusion_hidden', 'Success', 'Run Name']
    
    # Create table using simple formatting
    col_widths = [max(len(str(h)), max(len(str(row[i])) for row in table_data)) for i, h in enumerate(headers)]
    separator = '+' + '+'.join(['-' * (w + 2) for w in col_widths]) + '+'
    
    table = separator + '\n'
    table += '|' + '|'.join([f' {h:<{col_widths[i]}} ' for i, h in enumerate(headers)]) + '|\n'
    table += separator + '\n'
    for row in table_data:
        table += '|' + '|'.join([f' {str(cell):<{col_widths[i]}} ' for i, cell in enumerate(row)]) + '|\n'
    table += separator
    
    print("\n" + "="*80)
    print("Dimension Search Results")
    print("="*80)
    print(table)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(table)
        print(f"\nTable saved to: {output_file}")
    
    # Summary statistics
    total = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    failed = total - successful
    
    print(f"\nSummary: {successful}/{total} successful, {failed}/{total} failed")
    
    return table


def create_reg_search_table(results, output_file=None):
    """Create a table for regularization search results."""
    if not results:
        print("No results to display")
        return
    
    # Extract data
    table_data = []
    for r in results:
        row = [
            r.get('model_dropout', 'N/A'),
            r.get('weight_decay', 'N/A'),
            r.get('clip_norm', 'N/A'),
            r.get('fusion_dropout', 'N/A') if 'fusion_dropout' in r else '-',
            '✅' if r.get('success', False) else '❌',
            r.get('run_name', 'N/A')
        ]
        table_data.append(row)
    
    headers = ['model_dropout', 'weight_decay', 'clip_norm', 'fusion_dropout', 'Success', 'Run Name']
    
    # Create table using simple formatting
    col_widths = [max(len(str(h)), max(len(str(row[i])) for row in table_data)) for i, h in enumerate(headers)]
    separator = '+' + '+'.join(['-' * (w + 2) for w in col_widths]) + '+'
    
    table = separator + '\n'
    table += '|' + '|'.join([f' {h:<{col_widths[i]}} ' for i, h in enumerate(headers)]) + '|\n'
    table += separator + '\n'
    for row in table_data:
        table += '|' + '|'.join([f' {str(cell):<{col_widths[i]}} ' for i, cell in enumerate(row)]) + '|\n'
    table += separator
    
    print("\n" + "="*80)
    print("Regularization Search Results")
    print("="*80)
    print(table)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(table)
        print(f"\nTable saved to: {output_file}")
    
    # Summary statistics
    total = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    failed = total - successful
    
    print(f"\nSummary: {successful}/{total} successful, {failed}/{total} failed")
    
    return table


def create_latex_table(results, search_type='dim'):
    """Create LaTeX table format."""
    if search_type == 'dim':
        headers = [r'd\_model', r'd\_ff', r'sent\_hidden', r'fusion\_hidden', 'Success', 'Run Name']
        rows = []
        for r in results:
            rows.append([
                r.get('d_model', 'N/A'),
                r.get('d_ff', 'N/A'),
                r.get('sentiment_hidden_dim', 'N/A'),
                r.get('fusion_hidden_dim', 'N/A'),
                'Yes' if r.get('success', False) else 'No',
                r.get('run_name', 'N/A').replace('_', r'\_')
            ])
        caption = "Dimension Search Results"
    else:  # reg
        headers = [r'model\_dropout', r'weight\_decay', r'clip\_norm', r'fusion\_dropout', 'Success', 'Run Name']
        rows = []
        for r in results:
            rows.append([
                r.get('model_dropout', 'N/A'),
                r.get('weight_decay', 'N/A'),
                r.get('clip_norm', 'N/A'),
                r.get('fusion_dropout', 'N/A') if 'fusion_dropout' in r else '-',
                'Yes' if r.get('success', False) else 'No',
                r.get('run_name', 'N/A').replace('_', r'\_')
            ])
        caption = "Regularization Search Results"
    
    # Generate LaTeX
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\small\n"
    latex += "\\begin{tabular}{" + "l " * len(headers) + "}\n"
    latex += "\\toprule\n"
    latex += " & ".join([f"\\textbf{{{h}}}" for h in headers]) + " \\\\\n"
    latex += "\\midrule\n"
    
    for row in rows:
        latex += " & ".join([str(x) for x in row]) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += "\\end{table}\n"
    
    return latex


def main():
    parser = argparse.ArgumentParser(description='Visualize hyperparameter search results')
    parser.add_argument('json_file', type=str, help='Path to search results JSON file')
    parser.add_argument('--type', type=str, choices=['dim', 'reg', 'auto'],
                       default='auto', help='Search type (dim/reg/auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for table (text format)')
    parser.add_argument('--latex', type=str, default=None,
                       help='Output file for LaTeX table')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.json_file)
    
    # Auto-detect type
    if args.type == 'auto':
        if results and 'd_model' in results[0]:
            args.type = 'dim'
        elif results and 'model_dropout' in results[0]:
            args.type = 'reg'
        else:
            print("Could not auto-detect search type. Please specify --type")
            return
    
    # Create table
    if args.type == 'dim':
        create_dim_search_table(results, args.output)
    else:
        create_reg_search_table(results, args.output)
    
    # Create LaTeX if requested
    if args.latex:
        latex_table = create_latex_table(results, args.type)
        with open(args.latex, 'w') as f:
            f.write(latex_table)
        print(f"\nLaTeX table saved to: {args.latex}")


if __name__ == '__main__':
    main()


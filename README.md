# 📊 Automated Dataset Analysis & Reporting (CLI Tool, LLM-Powered)

This project offers a powerful command-line tool that automates the entire process of exploratory data analysis (EDA) for any tabular dataset. With a single command, it intelligently examines your data, generates Python code for analysis, executes it, and compiles the results into a polished Markdown report — complete with visualizations and expert-style commentary.

At the core of this tool is a Large Language Model (LLM), which drives every step of the pipeline: understanding dataset structure, selecting meaningful analyses, generating and refining code, and summarizing insights. It uses the OpenAI API to power its decisions, making your reports not only automated but also contextually smart.

Whether you're a data scientist, analyst, or engineer, this CLI-first solution is built to save you hours of manual work and deliver consistent, quality data reports — ready to present, or dive deeper.

---

## 🚀 Features

- **One-command analysis** – just run the script with your dataset and get a complete report.
- **Zero manual analysis** – just provide your dataset and get a full report.
- **LLM-driven pipeline** – intelligently understands your dataset and designs analysis steps.
- **Code generation loop** – automatically refines Python code until it runs correctly.
- **Visualization support** – generates plots and charts for clearer insights.
- **Markdown report** – outputs a clean, structured `.md` report ready for publishing or review.
- **Fully automated** – set it and forget it. One command to run the whole pipeline.

---

## 📦 How It Works

1. **Dataset Ingestion & Summary**:  
   The pipeline begins by loading the CSV file and extracting initial metrics and summaries to understand the structure and contents of the dataset.

2. **Analysis Planning via LLM**:  
   The extracted metadata is passed to an LLM to generate a list of potential analyses tailored to the dataset (e.g., correlation checks, distribution analysis, missing data evaluation, etc.).

3. **Python Code Generation**:  
   For each suggested analysis, the LLM generates Python code. A loopback system ensures that only syntactically and functionally valid code is accepted and executed.

4. **Result Collection**:  
   All the successfully executed analysis results (text and visualizations) are stored for final reporting.

5. **Report Generation**:  
   Using the accumulated results, the LLM creates a comprehensive Markdown (`.md`) report with explanations and charts.

---

## 🛠️ Installation & Setup

### 1. Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-folder>
```
### 2. Install dependencies:
```bash
pip install -r requirements.txt
```
### 🧠 Using OpenAI API
This project uses OpenAI's API for LLM-based tasks. You need an OpenAI API token to run the script.

### 📈 Run the Report Generator
```bash
python autolysis.py \
  --file <path_to_your_csv_file> \
  --limit-images <number_of_images_to_save> \
  --output <output_report_filename.md> \
  --token <your_api_token> \
  --save-token
```
## 📄 Output
A professional, data-driven report will be generated in Markdown format, containing:

Summary statistics

Key insights

Visualizations

LLM-generated commentary



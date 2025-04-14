# 📊 Automated Dataset Analysis & Reporting using LLMs

This project provides a fully automated pipeline to analyze any tabular dataset and generate a detailed, professional-grade report using a powerful Large Language Model (LLM). It leverages the OpenAI API to guide every step of the analysis — from understanding the dataset, determining insightful analysis steps, generating and refining code, to finally producing a Markdown report with visualizations.

---

## 🚀 Features

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
python generate_report.py \
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

Code snippets (if needed)


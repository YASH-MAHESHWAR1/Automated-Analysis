# /// script
# dependencies = [
#   "matplotlib>=3.9.3",
#   "numpy>=2.1.3",
#   "openai>=1.56.2",
#   "pandas>=2.2.3",
#   "pillow>=11.0.0",
#   "requests>=2.32.3",
#   "scikit-learn>=1.5.2",
#   "scipy>=1.14.1",
#   "seaborn>=0.13.2",
#   "statsmodels>=0.14.4",
# ]
# requires-python = ">=3.10"
# [tool]
# description = "Automated dataset analysis and reporting script leveraging LLMs."
# version = "0.1.0"
# ///
import argparse
import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import zscore, ttest_ind
import io
from PIL import Image
import os
import requests
import json
import sys
import traceback
from dotenv import load_dotenv

# Load environment variables for API key

AIPROXY_TOKEN = None
url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

headers = None
def load_token(cli_token=None, save_to_env=False):
    if cli_token:
        if save_to_env:
            with open("token.env", "w") as f:
                f.write(f"AIPROXY_TOKEN={cli_token}\n")
            print("Token saved to .env file.")
        return cli_token

    load_dotenv(dotenv_path='token.env')
    token = os.getenv("AIPROXY_TOKEN")
    if not token:
        print("Error: API token not found. Use --token or set AIPROXY_TOKEN in .env.")
        sys.exit(1)
    return token

def load_dataset(filename):
    """
    Load a dataset and return its data along with a preview.
    """
    try:
        data = pd.read_csv(filename, low_memory=False,encoding="ISO-8859-1")
        preview = data.head(3).to_string(index=False)
        return data, preview
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def generic_analysis(data):
    """
    Perform generic analyses on the dataset and return results.
    """
    results = {}
    numeric_data = data.select_dtypes(include=['number'])

    # Summary statistics
    results['Summary Statistics'] = data.describe(include='all').to_string()

    # Missing values
    results['Missing Values'] = data.isnull().sum().to_string()

    # Correlation matrix
    if numeric_data.shape[1] > 1:
        corr_matrix = numeric_data.corr()
        plt.figure(figsize=(5.12, 5.12), dpi=100)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix', fontsize=16)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.close()
        results['Correlation Matrix'] = corr_matrix.to_string()

    # T-Test
    if len(numeric_data.columns) >= 2:
        col1, col2 = numeric_data.columns[:2]
        stat, p_val = ttest_ind(numeric_data[col1], numeric_data[col2], nan_policy='omit')
        results['T-Test'] = f"Stat: {stat}, P-val: {p_val}"

    # Outliers detection
    if not numeric_data.empty:
        z_scores = numeric_data.apply(zscore)
        outliers = (z_scores.abs() > 3).sum()
        results['Outliers (Z-Score)'] = outliers.to_string()

        # Outliers histogram
        plt.figure(figsize=(5.12, 5.12), dpi=100)
        z_scores.abs().stack().hist(bins=30)
        plt.title('Outliers Distribution (Z-Scores)', fontsize=16)
        plt.xlabel('Z-Score', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.close()

    # Clustering
    if numeric_data.shape[0] > 5 and numeric_data.shape[1] > 1:
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), 
                                            columns=numeric_data.columns)

        # Apply KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        data['Cluster'] = kmeans.fit_predict(numeric_data_imputed)
        results['Clustering'] = data['Cluster'].value_counts().to_string()

        # Scatter plot for clusters
        if numeric_data_imputed.shape[1] >= 2:
            plt.figure(figsize=(5.12, 5.12), dpi=100)
            plt.scatter(numeric_data_imputed.iloc[:, 0], numeric_data_imputed.iloc[:, 1], 
                        c=data['Cluster'], cmap='viridis', s=50, alpha=0.7)
            plt.title('Cluster Visualization', fontsize=16)
            plt.xlabel(numeric_data_imputed.columns[0], fontsize=14)
            plt.ylabel(numeric_data_imputed.columns[1], fontsize=14)
            plt.colorbar(label='Cluster', pad=0.02)
            plt.grid(True)
            plt.tight_layout()
            plt.close()

    return results

def request_analysis_suggestions(data, generic_results):
    """
    Request analysis suggestions for a dataset using LLM, incorporating results from generic analyses.
    """
    columns = data.columns.tolist()
    rows_preview = data.head(3).to_string(index=False)
    dataset_info = f"Columns: {', '.join(columns)}\nFirst 3 rows:\n{rows_preview}"

    # Incorporate generic analysis results
    generic_analysis_summary = "\n".join(
        [f"- **{analysis}**:\n{result}" for analysis, result in generic_results.items()]
    )

    prompt = f"""
    you are an expert data analyst
    you have a dataset with the following structure:
    {dataset_info}
    
    The following generic analyses have already been performed on this dataset:
    {generic_analysis_summary}

    After thoroughly understanding the results of the generic analysis and carefully studying the dataset structure, provide a well-considered list of top 4 additional analyses that can be conducted to extract deeper insights and address specific questions about the data, rather than simply repeating or performing analyses without a clear purpose 
    Please provide a concise and well-thought-out list of analyses that will help uncover meaningful insights about the dataset.
    """

    messages = [{"role": "user", "content": prompt}]
    function_call = {
        "name": "extract_analysis_list",
        "description": "Extracts only the list of analysis suggestions.",
        "parameters": {
            "type": "object",
            "properties": {
                "analyses": {
                    "type": "array",
                    "description": "A list of suggested analyses based on the dataset and generic analysis results.",
                    "items": {"type": "string"}
                }
            },
            "required": ["analyses"]
        }
    }

    data_payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "functions": [function_call],
        "function_call": {"name": "extract_analysis_list"}
    }

    response = requests.post(url, headers=headers, json=data_payload)
    if response.status_code == 200:
        try:
            result = response.json()
            analyses = json.loads(result["choices"][0]["message"]["function_call"]["arguments"])["analyses"]
            return analyses
        except Exception as e:
            print(f"Error extracting analysis list: {e}")
            return []
    else:
        print(f"Error with LLM request: {response.status_code}")
        return []


def request_code_for_analysis(analysis, data_preview):
    """
    Request Python code for a specific analysis.
    """
    prompt = f"""
    You are an expert Python programer
    Please generate Python code to perform the following analysis:
    {analysis}
    
    The dataset is already provided as the variable `data`, which is a pandas DataFrame with the following structure:
    {data_preview}
    
    The code should:
    1. Use pandas, matplotlib, or seaborn as needed.
    2. Save maximum one generated chart as PNG files.
    3. diplay computed results or summaries which doesnot contains chart.Avoid displaying the charts
    **Important**: 
    - Return **only the Python code** to perform the analysis. 
    - Do not include any introductory statements or the word `Python` at the beginning of the response.
    - Only include the Python code itself.
    """
    function_call = {
        "name": "generate_code_for_analysis",
        "description": "Generates only Python code for the requested analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code for the analysis"
                }
            },
            "required": ["code"]
        }
    }

    data_payload = {
        "model": "gpt-4o-mini",  # Adjust model as needed
        "messages": [{"role": "user", "content": prompt}],
        "functions": [function_call],
        "function_call": {"name": "generate_code_for_analysis"}
    }

    response = requests.post(url, headers=headers, json=data_payload)
    if response.status_code == 200:
        try:
            result = response.json()
            code = json.loads(result["choices"][0]["message"]["function_call"]["arguments"])["code"]
            return code
        except Exception as e:
            print(f"Error extracting code: {e}")
            return None
    else:
        print(f"Error with LLM request: {response.status_code}")
        return None
def execute_generated_code_with_retry(code, globals_dict):
    """
    Execute generated Python code safely, retry with LLM corrections until successful.
    
    Parameters:
    - code: The initial Python code to execute.
    - globals_dict: The global dictionary context for code execution.
    - llm_correction_function: A function that takes the failed code and error message,
      and returns a corrected version of the code.
    """
    attempt = 0
    captured_output = io.StringIO()

    while attempt<5:
        attempt += 1
        sys.stdout = captured_output  # Redirect stdout to capture print statements

        try:
            # Attempt to execute the code
            exec(code, globals_dict)  # Execute the code in the given globals context
            sys.stdout = sys.__stdout__  # Reset stdout to the default (console)
            
            # Capture the successful output
            output = captured_output.getvalue()
            print(f"Code executed successfully on attempt {attempt}.")
            return output

        except Exception as e:
            # Capture the error message
            sys.stdout = sys.__stdout__  # Reset stdout to the default (console)
            error_message = traceback.format_exc()
            output = captured_output.getvalue() + f"\nError: {error_message}"
            print(f"Error on attempt {attempt}: {error_message}")

            # Request corrected code from the LLM
            corrected_code = llm_correction_function(code, error_message)

            if not corrected_code:
                # If the LLM fails to provide a corrected code, stop retrying
                print("LLM failed to provide corrected code. Aborting.")
                return output

            print(f"Retrying with corrected code (Attempt {attempt + 1})...")
            code = corrected_code
            print(code)
def llm_correction_function(code, error_message):
    """
    Request the LLM to correct the code based on the error message using function calling.
    """
    # Prepare the function definition for the LLM
    function_call = {
        "name": "generate_corrected_code",
        "description": "Generate corrected Python code for the provided analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The corrected Python code that addresses the issue."
                }
            },
            "required": ["code"]
        }
    }

    # Prepare the prompt
    prompt = f"""
    The following Python code failed to execute:
    ```
    {code}
    ```
    The error encountered was:
    ```
    {error_message}
    ```
    Please provide a corrected version of the code.
    **Important**: 
    - Return **only the Python code** to perform the analysis.
    """

    # Request corrected code from the LLM
    data_payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "functions": [function_call],
        "function_call": {"name": "generate_corrected_code"}
    }

    # Make the API request
    response = requests.post(url, headers=headers, json=data_payload)

    if response.status_code == 200:
        try:
            result = response.json()
            # Extract the corrected code from the function_call arguments
            corrected_code = json.loads(result["choices"][0]["message"]["function_call"]["arguments"])["code"]
            return corrected_code.strip()
        except Exception as e:
            print(f"Error extracting corrected code: {e}")
            return None
    else:
        print(f"LLM request failed with status code {response.status_code}.")
        return None

def request_report_based_on_analysis(data, analysis_results,filename):
    """
    Request a detailed narrative story from LLM based on the dataset and analyses performed.
    """
    prompt = f"""
    Please analyze the following dataset details and provide insights:

    ## Dataset Overview
    - **Filename**: {filename}
    - **Columns**: {', '.join(data.columns)}
    - **Data Types**: {', '.join([f"{col}: {data[col].dtype}" for col in data.columns])}

    Analyses performed with its result:
    """
    for i, (key, value) in enumerate(analysis_results.items()):
        prompt += f"""
        ### Analysis {i+1}: {key}
        - **Result**:
        {value}
        """

    
    prompt += f"""
    
    ## Instructions for the Narrative
    Please explain the dataset and the results of each analysis in detail:
    1. Summarize the dataset structure (columns, data types).
    2. Discuss the insights from each analysis result in detail, including descriptive statistics and correlations.
    3. Provide a cohesive conclusion summarizing the dataset and its analysis result.
    4. insert saved png images of the generated chart in teh report and describe the chart under its respective analysis 

    At the end, provide a summary of the dataset and its analyses result.
    """
    
    messages = [{"role": "user", "content": prompt}]
    data_payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 1500
    }
    
    response = requests.post(url, headers=headers, json=data_payload)
    if response.status_code == 200:
        try:
            result = response.json()
            report = result["choices"][0]["message"]["content"].strip()
            return report
        except Exception as e:
            print(f"Error generating report: {e}")
            return None
    else:
        print(f"Error with LLM request: {response.status_code}")
        return None  
def main(filename, image_limit, output_file, token, save_token_flag):
    global AIPROXY_TOKEN
    AIPROXY_TOKEN = load_token(token, save_to_env=save_token_flag)
    global headers
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    data, preview = load_dataset(filename)
    if data is None:
        return

    print("Dataset preview:\n")
    print(preview)
    generic_results = generic_analysis(data)
    print("\nRequesting analysis suggestions from LLM...")
    analysis_list = request_analysis_suggestions(data, generic_results)
    if not analysis_list:
        print("No analysis suggestions were provided.")
        return

    print("\nLLM suggested the following analyses:")
    for idx, analysis in enumerate(analysis_list, start=1):
        print(f"{idx}. {analysis}")

    original_savefig = plt.savefig
    image_save_count = 0

    def limited_savefig(*args, **kwargs):
        nonlocal image_save_count
        if image_save_count < image_limit:
            image_save_count += 1
            original_savefig(*args, **kwargs)
            image_path = args[0]
            if image_path.endswith(('.png', '.jpg', '.jpeg')):
                with Image.open(image_path) as img:
                    img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                    img.save(image_path, format="PNG", optimize=True)
                    print(f"Image resized and saved: {image_path}")
            else:
                print(f"Skipping resize for non-image file: {image_path}")
            print(f"Saved image {image_save_count}/{image_limit}: {image_path}")
        else:
            print("Image saving limit reached. Skipping save.")

    plt.savefig = limited_savefig

    final_result = {}
    for analysis in analysis_list:
        print(f"\nRequesting code for analysis: {analysis}")
        code = request_code_for_analysis(analysis, preview)
        if code:
            print(f"\nGenerated code for {analysis}:\n")
            print(code)
            print("\nExecuting code...")
            globals_dict = {"data": data, "pd": pd, "plt": plt, "sns": sns}
            success = execute_generated_code_with_retry(code, globals_dict)
            if success:
                final_result[analysis] = success
                print("---------------------------------")
                print(success)
                print("----------------------------------")
            else:
                print(f"Code execution failed for analysis: {analysis}")
        else:
            print(f"Failed to get code for analysis: {analysis}")

    plt.savefig = original_savefig

    report = request_report_based_on_analysis(data, final_result, filename)
    if report:
        print("\nDetailed Report:\n")
        print(report)
        with open(output_file, "w") as file:
            file.write(report)

        print(f"Report saved as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated dataset analysis and report generation.")
    parser.add_argument("--file", required=True, help="Path to the input dataset CSV file.")
    parser.add_argument("--limit-images", type=int, default=5, help="Maximum number of images to save (default: 5).")
    parser.add_argument("--output", default="README.md", help="Output report file (default: README.md).")
    parser.add_argument("--token", help="AIPROXY API token (overrides .env).")
    parser.add_argument("--save-token", action="store_true", help="Save the token to .env for future use.")

    args = parser.parse_args()
    main(args.file, args.limit_images, args.output, args.token, args.save_token)



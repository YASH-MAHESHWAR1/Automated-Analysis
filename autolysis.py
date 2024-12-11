# /// script
# dependencies = [
#   "matplotlib>=3.9.3",
#   "numpy>=2.1.3",
#   "openai>=1.56.2",
#   "pandas>=2.2.3",
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



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import io
from PIL import Image
import os
import requests
import json
import sys
# Load environment variables for API key
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {AIPROXY_TOKEN}",
    "Content-Type": "application/json"
}

def load_dataset(filename):
    """
    Load a dataset and return its data along with a preview.
    """
    try:
        data = pd.read_csv(filename, encoding="ISO-8859-1")
        preview = data.head(3).to_string(index=False)
        return data, preview
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

def generic_analysis(data):
    """
    Perform generic analyses that apply to most datasets.
    """
    results = {}
    print("Performing generic analyses...")

    # Summary statistics
    try:
        summary_stats = data.describe(include='all').to_string()
        results['Summary Statistics'] = summary_stats
        print("Summary Statistics:\n", summary_stats)
    except Exception as e:
        print(f"Error generating summary statistics: {e}")

    # Missing values count
    try:
        missing_values = data.isnull().sum().to_string()
        results['Missing Values'] = missing_values
        print("Missing Values:\n", missing_values)
    except Exception as e:
        print(f"Error counting missing values: {e}")

    # Correlation matrix
    try:
        numeric_data = data.select_dtypes(include=['number'])
        if numeric_data.shape[1] > 1:
            corr_matrix = numeric_data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.show()  # Display the plot
            print(corr_matrix.to_string())
            results['Correlation Matrix'] = corr_matrix.to_string()
            print("Correlation Matrix displayed.")
        else:
            print("Insufficient numerical columns for correlation matrix.")
    except Exception as e:
        print(f"Error generating correlation matrix: {e}")

    # Outlier detection (z-scores)
    try:
        numeric_data = data.select_dtypes(include=['number'])
        z_scores = (numeric_data - numeric_data.mean()) / numeric_data.std()
        outliers = (z_scores.abs() > 3).sum().to_string()
        results['Outliers (Z-Score)'] = outliers
        print("Outliers detected using Z-Scores:\n", outliers)
    except Exception as e:
        print(f"Error detecting outliers: {e}")

    # Clustering
    try:
        numeric_data = data.select_dtypes(include=['number']).dropna()
        if numeric_data.shape[0] > 5 and numeric_data.shape[1] > 1:
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(numeric_data)  # Perform clustering
            data.loc[numeric_data.index, 'Cluster'] = clusters  # Assign labels to the original rows in 'data'
            results['Clustering'] = data['Cluster'].value_counts().to_string()
            print("Clustering results:\n", results['Clustering'])
        else:
            print("Insufficient data for clustering.")
    except Exception as e:
        print(f"Error performing clustering: {e}")

    # Hierarchical clustering
    try:
        numeric_data = data.select_dtypes(include=['number']).dropna()
        if numeric_data.shape[0] > 5:
            linked = linkage(numeric_data, method='ward')
            plt.figure(figsize=(10, 7))
            dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
            plt.title('Hierarchical Clustering Dendrogram')
            print("Hierarchical clustering dendrogram displayed.")
        else:
            print("Insufficient data for hierarchical clustering.")
    except Exception as e:
        print(f"Error performing hierarchical clustering: {e}")

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

    Based on the dataset structure and the findings from the generic analyses, provide a list of further analyses that should be conducted to extract deeper insights or answer potential questions about the data. 
   
    Your suggestions should:
    - Be tailored to the specific dataset and its characteristics.
    - Take into account the results from the generic analyses to avoid redundancy.
    - Focus on identifying patterns, relationships, or other significant features in the dataset.
    - Include advanced statistical or machine learning methods if relevant.
    
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
    3. diplay computed results or summaries which doesnot contains chart.
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

    Returns:
    - A tuple (success, output, final_code):
      - success: True if code executed successfully, False otherwise.
      - output: The captured output of the code execution (stdout or error messages).
      - final_code: The final corrected version of the code that was executed.
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
            error_message = str(e)
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
def main(filename):
    # Step 1: Load dataset
    data, preview = load_dataset(filename)
    if data is None:
        return

    print("Dataset preview:\n")
    print(preview)
    generic_results = generic_analysis(data)
    # Step 2: Get analysis suggestions
    print("\nRequesting analysis suggestions from LLM...")
    analysis_list = request_analysis_suggestions(data,generic_results)
    if not analysis_list:
        print("No analysis suggestions were provided.")
        return

    print("\nLLM suggested the following analyses:")
    for idx, analysis in enumerate(analysis_list, start=1):
        print(f"{idx}. {analysis}")

    # Step 3: Prepare for PNG file-saving limit
    original_savefig = plt.savefig  # Save the original function
    image_save_count = 0  # Counter for saved images
    max_images = 5  # Limit the number of PNG files

    def limited_savefig(*args, **kwargs):
        """
        Limit the number of saved images to `max_images`, resize to 512x512 if necessary,
        and save the image using `plt.savefig`.
        """
        nonlocal image_save_count  # Use global to modify the counter
        if image_save_count < max_images:
            image_save_count += 1
        
            # Call the original savefig to save the image
            original_savefig(*args, **kwargs)
            
            # After saving the image, resize it if needed
            image_path = args[0]
            if image_path.endswith(('.png', '.jpg', '.jpeg')):
                # Open the image using Pillow to resize it
                with Image.open(image_path) as img:
                    img.thumbnail((512, 512), Image.Resampling.LANCZOS)  # Resize to max 512x512, maintaining aspect ratio
                    img.save(image_path, format="PNG", optimize=True)
                    print(f"Image resized and saved: {image_path}")
            else:
                # If it's not an image file, skip resizing
                print(f"Skipping resize for non-image file: {image_path}")
            
            print(f"Saved image {image_save_count}/{max_images}: {image_path}")
        else:
            print("Image saving limit reached. Skipping save.")


    # Temporarily replace plt.savefig
    plt.savefig = limited_savefig

    # Step 4: Request and execute code for each analysis
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
                # diffrentiating the real ouput of the analysis
                print("---------------------------------")
                print(success)
                print("----------------------------------")
            else:
                print(f"Code execution failed for analysis: {analysis}")
        else:
            print(f"Failed to get code for analysis: {analysis}")

    # Restore the original plt.savefig
    plt.savefig = original_savefig

    # Step 5: Request a detailed report
    report = request_report_based_on_analysis(data, final_result, filename)
    if report:
        print("\nDetailed Report:\n")
        print(report)
        with open("README.md", "w") as file:
            file.write(report)

    print("Report saved as README.md")



if __name__ == "__main__":
    # Get the dataset filename as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    
    filename = sys.argv[1]  # Get the dataset filename from the command-line argument
    main(filename)

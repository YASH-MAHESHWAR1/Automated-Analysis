# /// script
# dependencies = [
#   "matplotlib>=3.9.3",
#   "numpy>=2.1.3",
#   "openai>=1.56.2",
#   "pandas>=2.2.3",
#   "requests>=2.32.3",
#   "scikit-learn>=1.5.2",
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
import io
from PIL import Image
import os
import requests
import json
#from dotenv import load_dotenv
import sys

# Load environment variables for API key
#load_dotenv(dotenv_path='token.env')

#AIPROXY_TOKEN = os.getenv("API_TOKEN")
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

def request_analysis_suggestions(data):
    """
    Request analysis suggestions for a dataset using LLM with function calling.
    """
    columns = data.columns.tolist()
    rows_preview = data.head(3).to_string(index=False)
    dataset_info = f"Columns: {', '.join(columns)}\nFirst 3 rows:\n{rows_preview}"

    prompt = f"""
    I have a dataset with the following structure:
    {dataset_info}
    
    Suggest a list of analyses that would extract meaningful insights from the data.
    Out of these analyses, please ensure that only 4 focus on generating charts or visualizations.
    The rest of the analyses should be related to other forms of insight extraction (e.g., statistical tests, correlations, etc.).
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
                    "description": "A list of suggested analyses",
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
    Request a detailed report from LLM based on the dataset and analyses performed.
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
    2. Discuss the insights from each analysis result, including descriptive statistics and correlations.
    3. Provide a cohesive conclusion summarizing the dataset and its analyses.
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

    # Step 2: Get analysis suggestions
    print("\nRequesting analysis suggestions from LLM...")
    analysis_list = request_analysis_suggestions(data)
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

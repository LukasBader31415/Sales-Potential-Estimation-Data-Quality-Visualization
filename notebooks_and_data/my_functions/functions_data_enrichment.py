#import tiktoken
import pandas as pd
#from io import StringIO
import json
import os
#import time
import pickle
#from datetime import datetime



class NaicsProcessor:
    @staticmethod
    def expand_naics_and_split_value(df_occupation_industry, occu_prioritization_path):
        # Predefined mapping of old NAICS codes to new NAICS codes and their titles
        naics_mapping = {
            '3330A1': [
                {'naics_new': '3331', 'NAICS_TITLE': 'Agriculture, Construction, and Mining Machinery Manufacturing'},
                {'naics_new': '3332', 'NAICS_TITLE': 'Industrial Machinery Manufacturing'},
                {'naics_new': '3334', 'NAICS_TITLE': 'Ventilation, Heating, Air-Conditioning, and Commercial Refrigeration Equipment Manufacturing'},
                {'naics_new': '3339', 'NAICS_TITLE': 'Other General Purpose Machinery Manufacturing'}
            ],
            '3320A2': [
                {'naics_new': '3323', 'NAICS_TITLE': 'Architectural and Structural Metals Manufacturing'},
                {'naics_new': '3324', 'NAICS_TITLE': 'Boiler, Tank, and Shipping Container Manufacturing'}
            ],
            '3320A1': [
                {'naics_new': '3321', 'NAICS_TITLE': 'Forging and Stamping'},
                {'naics_new': '3322', 'NAICS_TITLE': 'Cutlery and Handtool Manufacturing'},
                {'naics_new': '3325', 'NAICS_TITLE': 'Hardware Manufacturing'},
                {'naics_new': '3326', 'NAICS_TITLE': 'Spring and Wire Product Manufacturing'},
                {'naics_new': '3329', 'NAICS_TITLE': 'Other Fabricated Metal Product Manufacturing'}
            ],
            '3250A1': [
                {'naics_new': '3251', 'NAICS_TITLE': 'Basic Chemical Manufacturing'},
                {'naics_new': '3252', 'NAICS_TITLE': 'Resin, Synthetic Rubber, and Artificial and Synthetic Fibers and Filaments Manufacturing'},
                {'naics_new': '3253', 'NAICS_TITLE': 'Pesticide, Fertilizer, and Other Agricultural Chemical Manufacturing'},
                {'naics_new': '3259', 'NAICS_TITLE': 'Other Chemical Product and Preparation Manufacturing'}
            ],
            '3370A1': [
                {'naics_new': '3371', 'NAICS_TITLE': 'Household and Institutional Furniture and Kitchen Cabinet Manufacturing'},
                {'naics_new': '3372', 'NAICS_TITLE': 'Office Furniture (including Fixtures) Manufacturing'}
            ]
        }
        
        # Step 1: Read prioritization data from the Excel file
        occu_prioritization = pd.read_excel(occu_prioritization_path, sheet_name='priorisierung')
        
        # Step 2: Merge the occupation-industry DataFrame with prioritization data
        df_occupation_industry_2 = df_occupation_industry.merge(
            occu_prioritization[['OCC_CODE', 'Prio']], on='OCC_CODE', how='left'
        )
        
        # Filter rows with prioritization levels 2 or 3
        df_occupation_industry_2_filtered = df_occupation_industry_2[df_occupation_industry_2['Prio'].isin([2, 3])]
        
        # Step 3: Expand NAICS codes based on the predefined mapping
        expanded_rows = []
        
        for old_code, mappings in naics_mapping.items():
            rows_to_expand = df_occupation_industry_2_filtered[df_occupation_industry_2_filtered['NAICS_CODE'] == old_code]
            
            if not rows_to_expand.empty:
                for occ_code in rows_to_expand['OCC_CODE'].unique():
                    occ_rows = rows_to_expand[rows_to_expand['OCC_CODE'] == occ_code]
                    # Distribute the 'emp_occupation' value equally among the new NAICS codes
                    value_per_code = occ_rows['emp_occupation'].sum() / len(mappings)
                    
                    for mapping in mappings:
                        temp_df = occ_rows.copy()
                        temp_df['NAICS_CODE'] = mapping['naics_new']
                        temp_df['NAICS_TITLE'] = mapping['NAICS_TITLE']
                        temp_df['emp_occupation'] = value_per_code
                        expanded_rows.append(temp_df)
        
        # Combine expanded rows into a DataFrame
        expanded_df = pd.concat(expanded_rows, ignore_index=True)
        
        # Final DataFrame: Include both original and expanded rows
        result_df = pd.concat(
            [df_occupation_industry_2_filtered[~df_occupation_industry_2_filtered['NAICS_CODE'].isin(naics_mapping.keys())], 
             expanded_df],
            ignore_index=True
        )
        
        return result_df


class JobInformationProcessor:
    def __init__(self, dataframe):
        """
        Initializes the JobInformationProcessor with a DataFrame containing the necessary job data.
        
        Parameters:
        - dataframe: A pandas DataFrame containing the job and industry data.
        """
        self.df = dataframe
        self.tasks_job = []

    def generate_job_information_prompt(self, occ_code, occ_title, naics, naics_title, number):
        """
        Generates a prompt to determine the field of activity based on the provided job information.
        
        Parameters:
        - occ_code: SOC code of the occupation
        - occ_title: Title of the occupation
        - naics: NAICS code of the industry
        - naics_title: Title of the NAICS industry
        - number: Number of activities to describe

        Returns:
        - The prompt as a string
        """
        prompt = f"""
        You are a labor market expert and analyst.
        What is the field of activity of the occupation with the SOC code {occ_code} with the job title {occ_title} in the industry {naics_title} with the NAICS code {naics}.
        Please focus on the top {number} most common Activities that are industry-specific in this job.
        
        Second, describe a precise "job description" in a maximum of 15 words.
        Output the result in JSON format as a list of objects, like this:

        [
            {{
                "Activities": "Main activity 1", "Main activity 2", "Main activity 3", ...
                "Job_Description": "Precise description in 15 words",
            }},
        ]
        """
        return prompt

        
    def create_job_task(self, occ_code, occ_title, naics, naics_title, number, index):
        """
        Creates a task for the API request based on the job data.
        
        Parameters:
        - occ_code: SOC code of the occupation
        - occ_title: Title of the occupation
        - naics: NAICS code of the industry
        - naics_title: Title of the NAICS industry
        - number: Number of activities to describe
        - index: The index of the current row in the DataFrame

        Returns:
        - A task for the API request
        """
        prompt_content = self.generate_job_information_prompt(occ_code, occ_title, naics, naics_title, number)
        
        task = {
            "custom_id": f"job-task-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",  # Model version
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": prompt_content
                    }
                ]
            }
        }
        return task

  
    def generate_job_tasks(self, task_file_path):
        """
        Generates a list of tasks based on the job data in the DataFrame
        and writes them directly to a JSON Lines file.
    
        Parameters:
        - task_file_path: The file path where the tasks should be saved.
    
        Returns:
        - A list of the generated tasks.
        """
        # Open the specified file for writing
        with open(task_file_path, 'w') as file:
            # Iterate over each row in the DataFrame
            for index, row in self.df.iterrows():
                # Extract relevant job data
                occ_code = row['OCC_CODE']
                occ_title = row['OCC_TITLE']
                naics = row['NAICS_CODE']
                naics_title = row['NAICS_TITLE']
                number = 3  # Number of activities to describe
                
                # Create a task using the provided data
                task = self.create_job_task(occ_code, occ_title, naics, naics_title, number, index)
                
                # Add the task to the internal task list
                self.tasks_job.append(task)
                
                # Write the task to the file in JSON format
                file.write(json.dumps(task) + '\n')
        
        # Confirm the tasks have been saved
        print(f"Tasks successfully saved to {task_file_path}")
        
        # Return the list of generated tasks
        return self.tasks_job


class ToolConsumptionProcessor:
    def __init__(self, dataframe):
        """
        Initializes the JobInformationProcessor with a DataFrame containing the necessary job data.
        
        Parameters:
        - dataframe: A pandas DataFrame containing the job and industry data.
        """
        self.df = dataframe
        self.tasks_tools = []


    def generate_tool_consumption_prompt(self, OCC_CODE, OCC_TITLE, NAICS, NAICS_Kennung, Job_Description, Activities):        
            prompt = f'''
            You are a Tool expert. 
            Your goal is to evaluate the consumption of special tools on the basis of various information with one number between 0 (no consumption) and 10 (high consumption)
            The focus here is on the consumption of an employee with the following profile:
            - Occupation: SOC-CODE: {OCC_CODE}, SOC-TITLE: {OCC_TITLE}, 
            - Industry: NAICS-CODE: {NAICS}, NAICS-TITLE: {NAICS_Kennung},
            - Industry-specific fields of activities: {Activities} and work description {Job_Description}
            
            These are possible application examples of the tools and alternative names in order to better understand them:
            - [Tool 1]: Description of use of the product
            - [Tool 2]: Description of use of the product
            - [Tool 3]: Description of use of the product
            - [Tool 4]: Description of use of the product
            - [Tool 5]: Description of use of the product
        
            You will output a JSON object containing only one number to estimate the consumption per tool between 0 (no consumption) and 10 (high consumption)
            Here are guide values for classifying consumption.
            0: No consumption (tool is not used at all)
            1-3: Very low consumption (rarely or only used in exceptional cases)
            4-6: Moderate consumption (occasional use, but not daily)
            7-9: High consumption (regular use, daily)
            10: Permanent use (continuous and intensive use):
            
            {{
                consumption_tool_1: int[],
                consumption_tool_2: int[],
                consumption_tool_3: int[],
                consumption_tool_4: int[],
                consumption_tool_5: int[],
            }}
            Please make sure that you only output one value per tool consumption!
            '''
            return prompt

    def create_tool_consumption_task(self, OCC_CODE, OCC_TITLE, NAICS, NAICS_TITLE, Job_Description, Activities, index):
        """
        Creates a task for an API request based on tool consumption data.
    
        Parameters:
        - OCC_CODE: The code for the job category (SOC Code).
        - OCC_TITLE: The title of the job.
        - NAICS: The NAICS code of the industry.
        - NAICS_TITLE: The title of the NAICS industry.
        - Job_Description: A description of the job.
        - Activities: The activities performed in the job.
        - index: The index of the current row in the DataFrame.
    
        Returns:
        - A task dictionary for the API request.
        """
        # Generate the content for the API request using the given parameters
        prompt_content = self.generate_tool_consumption_prompt(
            OCC_CODE, OCC_TITLE, NAICS, NAICS_TITLE, Job_Description, Activities
        )
    
        # Define the API task as a dictionary
        task = {
            "custom_id": f"tool-consumption-task-{index}",  # Unique ID for the task
            "method": "POST",  # HTTP method for the API request
            "url": "/v1/chat/completions",  # API endpoint
            "body": {
                "model": "gpt-4o",  # AI model version to be used
                "temperature": 0,  # Control response variability
                "response_format": {"type": "json_object"},  # Specify response type
                "messages": [
                    {
                        "role": "system",  # Role of the message
                        "content": prompt_content  # Generated content for the API
                    }
                ]
            }
        }
    
        # Return the constructed API task
        return task

    def generate_tool_consumption_tasks(self, task_file_path):
        """
        Generates a list of tasks based on the tool consumption data in the DataFrame
        and writes them to a JSON Lines file.
    
        Parameters:
        - task_file_path: The file path where the tasks should be saved.
    
        Returns:
        - A list of the generated tasks.
        """
        # Open the specified file for writing tasks
        with open(task_file_path, 'w') as file:
            # Iterate over each row in the DataFrame
            for index, row in self.df.iterrows():
                # Extract relevant job data from the row
                OCC_CODE = row['OCC_CODE']
                OCC_TITLE = row['OCC_TITLE']
                NAICS = row['NAICS_CODE']
                NAICS_TITLE = row['NAICS_TITLE']
                Job_Description = row['Job_Description']
                Activities = row['Activities']
    
                # Create a task using the extracted data
                task = self.create_tool_consumption_task(
                    OCC_CODE, OCC_TITLE, NAICS, NAICS_TITLE, Job_Description, Activities, index
                )
    
                # Add the task to the internal list of tool-related tasks
                self.tasks_tools.append(task)
    
                # Write the task to the file in JSON Lines format
                file.write(json.dumps(task) + '\n')
    
        # Confirm that the tasks were saved successfully
        print(f"Tasks successfully saved to {task_file_path}")
    
        # Return the list of generated tasks
        return self.tasks_tools




class BatchProcessor:
    def __init__(self):
        """
        Initializes the BatchProcessor with the list of tasks.
        
        Parameters:
        - tasks: List of tasks to process in batch.
        """
        

    

    def submit_batch_job(self, client, task_file_name="batch_tasks.jsonl", batch_id_file="batch_id.txt"):
        """
        Submits the batch job to the client for processing. The task file will be uploaded,
        and the batch job will be created. Allows custom task file name and batch ID file name.
        
        Parameters:
        - client: The client used for API interaction (e.g., OpenAI client).
        - task_file_name: Name of the file where tasks will be saved (default: 'batch_tasks.jsonl').
        - batch_id_file: Name of the file where batch ID will be saved (default: 'batch_id.txt').
        
        Returns:
        - batch_id: The ID of the submitted batch job.
        """
        # Upload the batch task file
        batch_file = client.files.create(
            file=open(task_file_name, "rb"),
            purpose="batch"
        )

        # Create the batch job
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"  # Adjust completion window as needed
        )

        batch_id = batch_job.id
        print(f"Batch job submitted. Batch ID: {batch_id}")
        
        # Save batch_id to custom file
        self.save_batch_id(batch_id, batch_id_file)

        return batch_id
    
    def save_batch_id(self, batch_id, batch_id_file="batch_id.txt"):
        """
        Saves the batch_id to a custom text file to allow recovery in case of a crash.
        
        Parameters:
        - batch_id: The ID of the batch job.
        - batch_id_file: Name of the file where the batch_id will be saved (default: 'batch_id.txt').
        """
        with open(batch_id_file, 'w') as file:
            file.write(batch_id)
        print(f"Batch ID saved to {batch_id_file}")

    def load_batch_id(self, batch_id_file="batch_id.txt"):
        """
        Loads the batch_id from the saved file if it exists. Returns None if no file is found.
        
        Parameters:
        - batch_id_file: Name of the file where the batch_id is saved (default: 'batch_id.txt').
        
        Returns:
        - batch_id: The saved batch ID or None if no file exists.
        """
        if os.path.exists(batch_id_file):
            with open(batch_id_file, 'r') as file:
                batch_id = file.read().strip()
                print(f"Batch ID loaded from {batch_id_file}: {batch_id}")
                return batch_id
        else:
            print(f"No saved batch ID found in {batch_id_file}.")
            return None

    def check_batch_status(self, client, batch_id):
        """
        Checks the status of the batch job until it is complete.
        
        Parameters:
        - client: The client used for API interaction.
        - batch_id: The ID of the batch job to check.
        
        Returns:
        - batch_job: The batch job object with the latest status.
        """
        batch_job = client.batches.retrieve(batch_id)
        if batch_job.status == "completed":
            return batch_job
        elif batch_job.status == "failed":
            raise Exception("Batch job failed")
        else:
            print(f"Batch job status: {batch_job.status}. Job is still in progress.")
            return batch_job

    def download_results(self, client, batch_id, result_file_name="batch_results.jsonl"):
        """
        Downloads the results from the completed batch job and saves them to a file.
        Also returns the results as a variable (string or JSON).
        
        Parameters:
        - client: The client used for API interaction.
        - batch_id: The ID of the completed batch job.
        - result_file_name: Name of the file where results will be saved (default: 'batch_results.jsonl').
        
        Returns:
        - result_content: The content of the results as a variable (string or JSON).
        """
        # Retrieve the batch job status
        batch_job = self.check_batch_status(client, batch_id)
    
        result_content = None
        if batch_job.status == "completed":
            result_file_id = batch_job.output_file_id
            result_content = client.files.content(result_file_id).content.decode("utf-8")  # Decode as string
    
            # Save results to the specified file
            try:
                with open(result_file_name, 'w') as file:
                    file.write(result_content)
                print(f"Results saved to {result_file_name}")
            except Exception as e:
                print(f"Error saving results to file: {e}")
        else:
            print(f"Status: {batch_job.status}.")
    
        return result_content  # Return the results as a string (or JSON if desired)
        
    def calculate_costs(self, result_data, model_type="4o"):
        """
        Calculates the cost of the batch job based on token usage.
        
        Parameters:
        - result_data: String content from the batch job results (can be a JSONL string with multiple JSON objects).
        - model_type: The model used for the batch job (default: "4o").
        
        Returns:
        - cost: The calculated cost based on tokens used.
        """
        results = []
        
        try:
            # Falls 'result_data' ein String ist, splitte ihn in Zeilen
            if isinstance(result_data, str):
                lines = result_data.splitlines()  # Zeilenweise Aufteilung
                for line in lines:
                    try:
                        # Versuche, jede Zeile als JSON zu laden
                        json_object = json.loads(line.strip())
                        results.append(json_object)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in line: {line}. Error: {e}")
            elif not isinstance(result_data, list):
                print("Error: 'result_data' must be either a string or a list.")
                return None
    
        except Exception as e:
            print(f"Error processing input data: {e}")
            return None
    
        # Berechne die Gesamtzahl der genutzten Token
        try:
            prompt_tokens_sum = sum(task['response']['body']['usage']['prompt_tokens'] for task in results)
            completion_tokens_sum = sum(task['response']['body']['usage']['completion_tokens'] for task in results)
            total_tokens = prompt_tokens_sum + completion_tokens_sum
    
            print(f"Prompt Tokens: {prompt_tokens_sum}")
            print(f"Completion Tokens: {completion_tokens_sum}")
            print(f"Total Tokens: {total_tokens}")
        except KeyError as e:
            print(f"Error accessing tokens data: Missing key {e}")
            return None
    
        # Kosten basierend auf dem Modelltyp
        if model_type == "4o":
            # Berechne die Kosten pro Token für das Modell "4o"
            batch_costs_4o = ((prompt_tokens_sum * 1.25 + completion_tokens_sum * 5) / 1000000)
            print(f"Cost for model 4o: {batch_costs_4o} USD")
            return batch_costs_4o
        else:
            print("Unknown model type for cost calculation.")
            return None
    
    def process_batch(self, client,task_file_name, result_file_name="batch_results.jsonl", batch_id_file="batch_id.txt"):
        """
        Full batch processing: saves tasks, submits the batch job, and returns batch id for later processing.
        Allows custom file names for tasks, results, and batch ID storage.
        
        Parameters:
        - client: The client used for API interaction.
        - task_file_name: Name of the file where tasks will be saved (default: 'batch_tasks.jsonl').
        - result_file_name: Name of the file where results will be saved (default: 'batch_results.jsonl').
        - batch_id_file: Name of the file where the batch_id will be saved (default: 'batch_id.txt').
        
        Returns:
        - batch_id: The ID of the batch job for later status checking and result retrieval.
        """
                
        # Submit the batch job and get batch_id
        batch_id = self.submit_batch_job(client, task_file_name, batch_id_file)
        
        return batch_id


    def process_and_save_results_job(self, original_df, result_file_name, pickle_file_name="processed_results.pkl"):
        """
        Processes the results from the batch job, loads them from a JSONL file, 
        and saves them to a DataFrame, which is then saved as a pickle.
        
        Parameters:
        - original_df: The original DataFrame that contains the raw data (such as OCC_CODE, OCC_TITLE, etc.).
        - result_file_name: The name of the JSONL file that contains batch results.
        - pickle_file_name: The name of the pickle file to save the processed DataFrame (default: 'processed_results.pkl').
        
        Returns:
        - result_df: The updated DataFrame with processed results.
        """
        # Load the JSONL-file
        results = []
        try:
            with open(result_file_name, 'r') as file:
                for line in file:
                    try:
                        json_object = json.loads(line.strip())
                        results.append(json_object)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
        except FileNotFoundError as e:
            print(f"Error opening file {result_file_name}: {e}")
            return None

        # Initialize an empty list to store the processed data
        df_data = []
        
        # Iterate through the rows of the original DataFrame
        for i, row in original_df.iterrows():
            # Match the results with the corresponding row in the original DataFrame
            result = results[i] if i < len(results) else None
            
            if result and isinstance(result, dict) and "response" in result:
                try:
                    if result["response"]["status_code"] == 200:
                        content = result["response"]["body"]["choices"][0]["message"]["content"]
                        parsed_content = json.loads(content.strip())
                        activities = parsed_content.get("Activities", [])
                        job_description = parsed_content.get("Job_Description", "No description provided")
                    else:
                        activities = []
                        job_description = "Error: Status not 200"
                except (json.JSONDecodeError, KeyError) as e:
                    activities = []
                    job_description = f"Error parsing result: {str(e)}"
            else:
                activities = []
                job_description = "No result or error in batch"
            
            # Append the processed data to the df_data list
            df_data.append({
                "OCC_CODE": row["OCC_CODE"],
                "OCC_TITLE": row["OCC_TITLE"],
                "NAICS_CODE": row["NAICS_CODE"],
                "NAICS_TITLE": row["NAICS_TITLE"],
                "emp_occupation": row["emp_occupation"],
                "Prio": row["Prio"],
                "Activities": activities,
                "Job_Description": job_description
            })
        
        # Create a new DataFrame from the processed data
        result_df = pd.DataFrame(df_data)
        
        # Save the results as a pickle file for future use
        try:
            with open(pickle_file_name, 'wb') as f:
                pickle.dump(result_df, f)
            print(f"Processed DataFrame saved as pickle: {pickle_file_name}")
        except Exception as e:
            print(f"Error saving processed DataFrame to pickle: {e}")
        
        return result_df  # Return the updated DataFrame

    def process_and_save_results_tool_consumption(jsonl_file: str, input_dataframe: pd.DataFrame, pickle_file: str):
        """
        Reads a JSONL file, expands a DataFrame based on the JSON content, 
        and saves the resulting DataFrame as a Pickle file.
    
        :param jsonl_file: Path to the JSONL file to be processed.
        :param input_dataframe: DataFrame to be extended.
        :param pickle_file: Path to the Pickle file where results will be saved.
        """
        # Initialize an empty list to store JSON objects
        results = []
        
        # Read the JSONL file line by line and append the parsed content to results
        with open(jsonl_file, 'r') as file:
            for line in file:
                json_object = json.loads(line.strip())
                results.append(json_object)
        
        # Determine the number of choices in the first result
        n = len(results[0]['response']['body']['choices'])
        
        # Expand the input DataFrame by repeating rows 'n' times
        df_expanded = input_dataframe.loc[input_dataframe.index.repeat(n)].reset_index(drop=True)
        
        # Extract the content of the 'choices' field and append it to the consumption_data list
        consumption_data = []
        for result in results:
            for choice in result['response']['body']['choices']:
                content = json.loads(choice['message']['content'])
                consumption_data.append(content)
        
        # Convert the list of dictionaries into a DataFrame
        df_consumption = pd.DataFrame(consumption_data)
        
        # Add a new 'Index' column to the expanded DataFrame for tracking the repetitions
        repeat_index = list(range(n)) * (len(df_expanded) // n)
        df_expanded['Index'] = repeat_index
        
        # Combine the expanded DataFrame with the new consumption data
        df_expanded = pd.concat([df_expanded, df_consumption], axis=1)
        
        # Save the final DataFrame to a Pickle file
        df_expanded.to_pickle(pickle_file)
        
        print(f"Processed DataFrame has been saved to '{pickle_file}'.")
    
        # Return the final expanded DataFrame
        return df_expanded

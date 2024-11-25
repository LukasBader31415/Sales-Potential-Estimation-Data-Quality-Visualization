import pandas as pd


class LoadConsumptionData:
    def __init__(self, file_path):
        """Initializes the object and loads the data from the Pickle file."""
        self.file_path = file_path
        self.tool_consumption_data = self.load_data(file_path)
        self.tool_consumption_industry = self.group_by_industry(self.tool_consumption_data)
        self.tool_consumption_occupation = self.group_by_occupation(self.tool_consumption_data)
    
    def load_data(self, file_path):
        """Loads a DataFrame from a Pickle file."""
        return pd.read_pickle(file_path)
    
    def group_by_industry(self, data):
        """Groups the data by 'NAICS_CODE' and 'NAICS_TITLE' and calculates the mean of the last five columns."""
        grouped_data = data.groupby(['NAICS_CODE', 'NAICS_TITLE'])[data.columns[-5:]].mean().reset_index()
        return grouped_data
    
    def group_by_occupation(self, data):
        """Groups the data by 'OCC_CODE' and 'OCC_TITLE' and calculates the mean of the last five columns."""
        grouped_data = data.groupby(['OCC_CODE', 'OCC_TITLE'])[data.columns[-5:]].mean().reset_index()
        return grouped_data
    
    def process_tool_consumption_industry(self):
        """
        Processes the tool consumption data by filtering and mapping specific NAICS codes.
        Then, it groups the data by 'NAICS_CODE' and calculates the mean for numerical columns.
        """
        # NAICS codes to filter
        new_codes = [
            '3331', '3332', '3334', '3339',
            '3323', '3324',
            '3321', '3322', '3325', '3326', '3329',
            '3251', '3252', '3253', '3259',
            '3371', '3372'
        ]
    
        # Mapping of NAICS codes
        mapping = {
            '3331': ['3330A1'],
            '3332': ['3330A1'],
            '3334': ['3330A1'],
            '3339': ['3330A1'],
            '3323': ['3320A2'],
            '3324': ['3320A2'],
            '3321': ['3320A1'],
            '3322': ['3320A1'],
            '3325': ['3320A1'],
            '3326': ['3320A1'],
            '3329': ['3320A1'],
            '3251': ['3250A1'],
            '3252': ['3250A1'],
            '3253': ['3250A1'],
            '3259': ['3250A1'],
            '3371': ['3370A1'],
            '3372': ['3370A1']
        }
    
        # Function to replace NAICS codes
        def replace_naics_code(code):
            return mapping.get(code, [code])  # Returns the new code or the original code if no mapping exists
    
        # Filter data by selected NAICS codes
        filtered_df = self.tool_consumption_industry[self.tool_consumption_industry['NAICS_CODE'].isin(new_codes)]
        filtered_df_2 = self.tool_consumption_industry[~self.tool_consumption_industry['NAICS_CODE'].isin(new_codes)]
    
        # Replace NAICS codes with mapped values
        filtered_df['NAICS_CODE'] = filtered_df['NAICS_CODE'].apply(lambda x: replace_naics_code(x)[0])
    
        # Group the filtered data and calculate the mean (excluding 'NAICS_TITLE')
        grouped_df = filtered_df.groupby('NAICS_CODE').agg(
            lambda x: x.mean() if x.name != 'NAICS_TITLE' else list(x)
        ).reset_index()
    
        # Concatenate the filtered and grouped DataFrames
        final_df = pd.concat([filtered_df_2, grouped_df], ignore_index=True)
    
        return final_df


class OccupationDataProcessor:
    def __init__(self, pickle_path, tool_consumption_occupation):
        """
        Initializes the class with the required input parameters.

        :param pickle_path: Path to the Pickle file containing the cleaned occupation data
        :param tool_consumption_occupation: DataFrame with tool consumption data and corresponding OCC_CODES
        """
        self.pickle_path = pickle_path
        self.tool_consumption_occupation = tool_consumption_occupation

    def process_data(self):
        """
        Processes the occupation data and returns the finalized DataFrame.
        
        :return: DataFrame occupation_data_final
        """
        # Load the Pickle file
        occupation_data = pd.read_pickle(self.pickle_path)
        
        # Rename columns
        occupation_data.rename(columns={'naics': 'NAICS_CODE'}, inplace=True)
        
        # Filter by relevant occupations (from tool_consumption_occupation)
        prio_occupations = self.tool_consumption_occupation['OCC_CODE'].unique()
        occupation_data = occupation_data[occupation_data['OCC_CODE'].isin(prio_occupations)]
        
        # Group by NAICS_CODE and sum 'emp_occupation'
        naics_emp_relevant_occupations = occupation_data.groupby('NAICS_CODE')['emp_occupation'].sum().reset_index()
        
        # Select relevant columns
        selected_columns = ['FIPS', 'OCC_CODE', 'OCC_TITLE', 'emp_occupation']
        occupation_data_filtered = occupation_data[selected_columns]
        
        # Aggregate the data: sum of employment and unique FIPS
        result = occupation_data_filtered.groupby(['OCC_CODE', 'OCC_TITLE']).agg(
            emp_sum=('emp_occupation', 'sum'),            # Sum of employment
            unique_FIPS=('FIPS', 'nunique')               # Number of unique FIPS
        ).reset_index()

        # Merge with tool_consumption_occupation (add extra columns)
        occupation_data_final = result.merge(
            self.tool_consumption_occupation[['OCC_CODE'] + list(self.tool_consumption_occupation.columns[-5:])],
            on='OCC_CODE', how='left'
        )
        
        return occupation_data_final, occupation_data


class OccupationDataProcessor:
    def __init__(self, occupation_data_path, tool_consumption_occupation):
        """
        Initializes the class with the paths and data.
        :param occupation_data_path: Path to the Pickle file containing the Occupation data.
        :param tool_consumption_occupation: DataFrame of tool consumption data by occupation.
        """
        self.occupation_data = self.load_data(occupation_data_path)
        self.tool_consumption_occupation = tool_consumption_occupation

    def load_data(self, file_path):
        """Loads a DataFrame from a Pickle file."""
        return pd.read_pickle(file_path)

    def process_occupation_data(self):
        """
        Processes the occupation data based on the described workflow.
        :return: The processed DataFrame.
        """
        # Rename column for consistency
        self.occupation_data.rename(columns={'naics': 'NAICS_CODE'}, inplace=True)

        # Extract prioritized occupations
        prio_occupations = self.tool_consumption_occupation['OCC_CODE'].unique()

        # Filter the occupation data by relevant occupations
        filtered_data = self.occupation_data[self.occupation_data['OCC_CODE'].isin(prio_occupations)]

        # Calculate the relevant employment numbers by NAICS_CODE
        naics_emp_relevant_occupations = (
            filtered_data.groupby('NAICS_CODE')['emp_occupation'].sum().reset_index()
        )

        # Select specific columns
        selected_columns = ['FIPS', 'OCC_CODE', 'OCC_TITLE', 'emp_occupation']
        occupation_data_filtered = filtered_data[selected_columns]

        # Group the filtered data
        result = occupation_data_filtered.groupby(['OCC_CODE', 'OCC_TITLE']).agg(
            emp_sum=('emp_occupation', 'sum'),       # Sum of employment numbers
            unique_FIPS=('FIPS', 'nunique')          # Number of unique FIPS codes
        ).reset_index()

        # Merge with the tool consumption data by occupation
        final_data = result.merge(
            self.tool_consumption_occupation[['OCC_CODE'] + list(self.tool_consumption_occupation.columns[-5:])],
            on='OCC_CODE',
            how='left'
        )

        return final_data, naics_emp_relevant_occupations, filtered_data



class PatternDataProcessor:
    def __init__(self, pattern_data_path, tool_consumption_industry, naics_emp_relevant_occupations):
        """
        Initializes the class with the paths and data.
        :param pattern_data_path: Path to the Pickle file containing the Pattern data.
        :param tool_consumption_industry: DataFrame of tool consumption data by industry.
        :param naics_emp_relevant_occupations: DataFrame with employment numbers by NAICS codes.
        """
        self.pattern_data_path = pattern_data_path
        self.tool_consumption_industry = tool_consumption_industry
        self.naics_emp_relevant_occupations = naics_emp_relevant_occupations
        self.pattern_data = self.load_data(pattern_data_path)

    def load_data(self, file_path):
        """Loads a DataFrame from a Pickle file."""
        return pd.read_pickle(file_path)

    def add_zeros(self, code):
        """Adds leading zeros to a code to ensure it has a length of 5."""
        if len(code) == 3:
            return '00' + code
        elif len(code) == 4:
            return '0' + code
        elif len(code) == 1:
            return '0000' + code
        return code

    def process_pattern_data(self):
        """
        Processes the Pattern data based on the described workflow.
        :param naics_prio: List of prioritized NAICS codes.
        :return: The processed DataFrame.
        """
        naics_prio = ['1133', '2123', '2211', '2362', '2371', '2381', '2382', '2383',
                    '3231', '5413', '5617', '8111', '2131', '2373', '2379', '3119',
                    '3219', '3222', '3261', '3313', '3320A2', '3327', '3320A1',
                    '3330A1', '3364', '3366', '3370A1', '3391', '3399', '4881', '5321',
                    '5612', '5613', '8113', '3211', '3212', '3315', '3328', '3362',
                    '3363', '3262', '3335', '2111', '2121', '2212', '3241', '3250A1',
                    '3311', '3312', '3333', '3336', '3345', '3361', '4811', '4812',
                    '4862', '4882', '3314', '3344', '2122', '4861', '3132', '3365']

        # Rename columns
        self.pattern_data.rename(columns={'naics': 'NAICS_CODE', 'DESCRIPTION': 'NAICS_TITLE'}, inplace=True)
        
        # Format FIPS codes
        self.pattern_data['FIPS'] = self.pattern_data['FIPS'].astype(str)
        self.pattern_data['FIPS'] = self.pattern_data['FIPS'].apply(self.add_zeros)
        
        # Filter by prioritized NAICS codes
        filtered_data = self.pattern_data[self.pattern_data['NAICS_CODE'].isin(naics_prio)]
        
        # Merge with employment data
        merged_data = filtered_data.merge(
            self.naics_emp_relevant_occupations[['NAICS_CODE', 'emp_occupation']],
            on='NAICS_CODE',
            how='left'
        )
        
        # Select relevant columns
        selected_columns = ['FIPS', 'NAICS_CODE', 'NAICS_TITLE', 'emp_occupation', 'est']
        pattern_data_filtered = merged_data[selected_columns]
        
        # Group and aggregate data
        result = pattern_data_filtered.groupby(['NAICS_CODE', 'NAICS_TITLE']).agg(
            emp_sum=('emp_occupation', 'sum'),  # Sum of employment numbers
            est_sum=('est', 'sum'),             # Sum of estimate values
            unique_FIPS=('FIPS', 'nunique')     # Number of unique FIPS codes
        ).reset_index()
        
        # Merge with tool consumption data
        final_data = result.merge(
            self.tool_consumption_industry[['NAICS_CODE'] + list(self.tool_consumption_industry.columns[-5:])],
            on='NAICS_CODE',
            how='left'
        )
        
        return final_data, filtered_data


class Prioritization:
    def __init__(self, occupation_data_final, pattern_data_final):
        """
        Initializes the class with the provided DataFrames.
        :param occupation_data_final: DataFrame with data for prioritizing occupations.
        :param pattern_data_final: DataFrame with data for prioritizing NAICS patterns.
        """
        self.occupation_data_final = occupation_data_final.copy()
        self.pattern_data_final = pattern_data_final.copy()
    
    def rank_occupation_columns(self):
        """
        Performs ranking for specific columns in the occupation_data_final DataFrame:
        - 'emp_sum'
        - 'unique_FIPS'
        - All columns starting with 'consumption'.
        
        Returns:
        - DataFrame with added ranking columns.
        """
        result_copy = self.occupation_data_final

        # List of columns to rank
        columns_to_rank = ['emp_sum', 'unique_FIPS'] + [
            col for col in result_copy.columns if col.startswith('consumption')
        ]

        # Calculate rankings
        for column in columns_to_rank:
            rank_column_name = f'rank_{column}'
            result_copy[rank_column_name] = result_copy[column].rank(method='min', ascending=False).astype(int)
        
        return result_copy
    
    def rank_pattern_columns(self):
        """
        Performs ranking for specific columns in the pattern_data_final DataFrame:
        - 'emp_sum'
        - 'est_sum'
        - 'unique_FIPS'
        - All columns starting with 'consumption'.
        
        Returns:
        - DataFrame with added ranking columns.
        """
        result_copy = self.pattern_data_final

        # List of columns to rank
        columns_to_rank = ['emp_sum', 'est_sum', 'unique_FIPS'] + [
            col for col in result_copy.columns if col.startswith('consumption')
        ]

        # Calculate rankings
        for column in columns_to_rank:
            rank_column_name = f'rank_{column}'
            result_copy[rank_column_name] = result_copy[column].rank(method='min', ascending=False).astype(int)
        
        return result_copy

    def calculate_weighted_rank_sum(self, df, weight_dict):
        """
        Calculates the weighted sum of the rank columns in a DataFrame.
        
        Parameters:
        - df: The DataFrame containing the rank columns.
        - weight_dict: A dictionary specifying the weights for each rank column.
        
        Returns:
        - A DataFrame with a new column 'weighted_rank_sum' containing the weighted sum.
        """
        # Calculate the weighted sum
        weighted_sum = 0
        for rank_column, weight in weight_dict.items():
            weighted_sum += df[rank_column] * weight

        # Add the weighted sum to the DataFrame
        df['weighted_rank_sum'] = weighted_sum
        df_sorted = df.sort_values(by='weighted_rank_sum', ascending=True)
        return df_sorted


class MasterDFBuilder:
    def __init__(self, original_occupation_df, original_pattern_df, county_information):
        """
        Initializes the class with the required DataFrames.

        :param original_occupation_df: DataFrame containing OCC_CODE and emp_occupation data
        :param original_pattern_df: DataFrame containing NAICS_CODE, emp, and est data
        :param county_information: DataFrame containing FIPS data
        """
        self.original_occupation_df = original_occupation_df
        self.original_pattern_df = original_pattern_df
        self.county_information = county_information
        self.master_df = pd.DataFrame(county_information['FIPS'].unique(), columns=['FIPS'])

    def _process_occ(self, occ_codes):
        """
        Processes the OCC_CODES and returns a list of DataFrames, one for each OCC_CODE.
        """
        df_list = []
        for code in occ_codes:
            filtered = self.original_occupation_df[self.original_occupation_df['OCC_CODE'] == code]
            occ_code = filtered['OCC_CODE'].unique()[0]
            aggregated = (
                filtered.groupby(['FIPS', 'OCC_CODE'])
                .agg(total_emp_occu=('emp_occupation', 'sum'))
                .reset_index()
            )
            aggregated.columns = ['FIPS', 'OCC_CODE', f'total_emp_occu_{occ_code}']
            df_list.append(aggregated)
        return df_list

    def _process_naics(self, naics_codes):
        """
        Processes the NAICS_CODES and returns a list of DataFrames, one for each NAICS_CODE.
        """
        df_list = []
        for code in naics_codes:
            filtered = self.original_pattern_df[self.original_pattern_df['NAICS_CODE'] == code]
            naics_code = filtered['NAICS_CODE'].unique()[0]
            aggregated = (
                filtered.groupby(['FIPS', 'NAICS_CODE'])
                .agg(total_emp_naics=('emp', 'sum'), total_est_naics=('est', 'sum'))
                .reset_index()
            )
            aggregated.columns = [
                'FIPS',
                'NAICS_CODE',
                f'total_emp_naics_{naics_code}',
                f'total_est_naics_{naics_code}',
            ]
            df_list.append(aggregated)
        return df_list

    def build_master_df(self, occ_top10, occ_top10_20, naics_top6, naics_top_metall, save_path):
        """
        Builds the master DataFrame and saves it directly as a Pickle file.
    
        :param occ_top10: List of the top 10 OCC_CODES
        :param occ_top10_20: List of the top 10-20 OCC_CODES
        :param naics_top6: List of the top 6 NAICS_CODES
        :param naics_top_metall: List of metal-related NAICS_CODES
        :param save_path: File path where the Pickle should be saved
        """
        # Process OCC_TOP10
        df_list_occ_top10 = self._process_occ(occ_top10)
        for occ_df in df_list_occ_top10:
            value_column = occ_df.columns[2]
            if value_column not in self.master_df.columns:
                self.master_df = self.master_df.merge(occ_df[['FIPS', value_column]], on='FIPS', how='left')
    
        # Process OCC_TOP10_20
        df_list_occ_top10_20 = self._process_occ(occ_top10_20)
        for occ_df in df_list_occ_top10_20:
            value_column = occ_df.columns[2]
            if value_column not in self.master_df.columns:
                self.master_df = self.master_df.merge(occ_df[['FIPS', value_column]], on='FIPS', how='left')
    
        # Process NAICS_TOP6
        df_list_naics_top6 = self._process_naics(naics_top6)
        for naics_df in df_list_naics_top6:
            if naics_df.columns[2] not in self.master_df.columns and naics_df.columns[3] not in self.master_df.columns:
                self.master_df = self.master_df.merge(naics_df[['FIPS', naics_df.columns[2], naics_df.columns[3]]], on='FIPS', how='left')
    
        # Process NAICS_TOP_METALL
        df_list_naics_top_metall = self._process_naics(naics_top_metall)
        for naics_df in df_list_naics_top_metall:
            if naics_df.columns[2] not in self.master_df.columns and naics_df.columns[3] not in self.master_df.columns:
                self.master_df = self.master_df.merge(naics_df[['FIPS', naics_df.columns[2], naics_df.columns[3]]], on='FIPS', how='left')
    
        # Fill all NaN values with 0
        self.master_df = self.master_df.fillna(0)
    
        # Save master_df as a Pickle file
        self.master_df.to_pickle(save_path)
        print(f"master_df successfully saved as a Pickle file at '{save_path}'")
    
        return self.master_df


class FeatureAggregator:
    def __init__(self, df, save_path):
        """
        Constructor to initialize the DataFrame and the path for saving the Pickle file.
        
        :param df: DataFrame to be aggregated
        :param save_path: The path where the Pickle file will be saved
        """
        self.df = df
        self.save_path = save_path

    def aggregate_columns(self, aggregate_columns):
        """
        Aggregates the specified columns in the DataFrame and saves the result as a Pickle file.

        :param aggregate_columns: A dictionary containing columns and aggregation functions
        :return: A DataFrame with the aggregated data
        """
        # New DataFrame with aggregated columns
        aggregated_data = {}
        
        # Perform aggregation on all columns
        for new_column, (columns, agg_func) in aggregate_columns.items():
            # Select relevant columns
            selected_columns_agg = self.df[columns]
            
            # Aggregate the columns with the desired function
            if agg_func == 'sum':
                aggregated_data[new_column] = selected_columns_agg.sum(axis=1)
            else:
                raise ValueError(f"Unsupported aggregation function: {agg_func}")
        
        # Combine the new aggregated columns into a DataFrame
        aggregated_df = pd.DataFrame(aggregated_data)
        
        # Merge the original DataFrame with the aggregated data
        result_df = pd.concat([self.df, aggregated_df], axis=1)
        
        # Select the first 11 and the last 5 columns from the resulting DataFrame
        selected_columns_result_df = result_df.columns[:11].to_list() + result_df.columns[-5:].to_list()
        result_df = result_df[selected_columns_result_df]
        
        # Save the DataFrame as a Pickle file
        result_df.to_pickle(self.save_path)
        print(f"result_df successfully saved as a Pickle file at '{self.save_path}'")
        
        return result_df

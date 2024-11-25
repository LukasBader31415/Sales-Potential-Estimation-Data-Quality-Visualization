import pandas as pd

class DataFrameProcessor:
    @staticmethod
    def replace_nulls_with_small_values(df):
        """
        Replaces null values in a DataFrame with a small value based on the smallest non-null value in each column.
        Also outputs the regular smallest non-null value of each column.

        Parameters:
            df (pd.DataFrame): The DataFrame with the columns to process.

        Returns:
            pd.DataFrame: DataFrame with replaced null values.
            dict: A dictionary with replacement values for each column.
            dict: A dictionary with the regular smallest non-null values for each column.
        """
        replacement_values = {}
        min_values = {}

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):  # Only process numeric columns
                # Find the smallest non-null value
                min_value = df[col][df[col] > 0].min()
                min_values[col] = min_value if min_value is not None else "No positive values"

                # Determine the replacement value
                replacement_value = 0.1
                while min_value is not None and min_value < replacement_value:
                    replacement_value /= 10  # Decrease the replacement value by an order of magnitude

                # Replace null values and store the replacement value
                df[col] = df[col].replace(0, replacement_value)
                replacement_values[col] = replacement_value

        #print("Column Overview:")
        #for column in df.columns:
            #print(f"{column}:")
            #print(f"  Regular smallest value: {min_values[column]}")
            #print(f"  Replacement value for nulls: {replacement_values.get(column, 'No nulls')}")

        return df, replacement_values, min_values
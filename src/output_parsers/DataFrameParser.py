import pandas as pd

class DataFrameParser:



    
    
    @staticmethod
    def transform_colums(df, column_list, pattern, transform_func, new_col_suffix="_parsed"):
        """
        Applies a regex pattern to specified columns and processes extracted data with a transformation function.

        Parameters:
        - df (pd.DataFrame): The input dataframe containing the columns to be processed.
        - column_list (list): A list of column names to search and parse using the regex pattern.
        - pattern (str): The regex pattern to extract relevant data from the columns.
        - transform_func (function): A function that takes extracted results and processes them.

        Returns:
        - pd.DataFrame: The original dataframe with new columns containing processed results.
        """
        for col in column_list:
            # Verify the column exists in the dataframe
            if col not in df.columns:
                print(f"Column '{col}' not found in the dataframe. Skipping.")
                continue

            # Create the new column name with a "_parsed" suffix
            new_col_name = f"{col}{new_col_suffix}"
            
            # Extract the pattern-matching content from the column
            extracted_series = df[col].astype(str).str.findall(pattern)
            
            # Apply the transformation function to the extracted data
            df[new_col_name] = extracted_series.apply(transform_func)
        return df
    

    @staticmethod
    def extract_answer(df, column_list):
        """
        Extracts text between <answer> and </answer> tags from specified columns.

        Parameters:
        - df (pd.DataFrame): The input dataframe containing the columns to be parsed.
        - column_list (list): A list of column names to search and extract answers from.

        Returns:
        - pd.DataFrame: The original dataframe with new columns containing extracted answers.
        """
        answer_pattern = r'<answer>(.*?)</answer>|\(answer>(.*)<\\/answer>|\(answer\)(.*)\(answer\)|\(answer\)(.*)<\|eot_id\|>|\(Answer\)(.*)<\|eot_id\|>|\(answer>(.*)<\|eot_id\|>|<Solution><Answer>(.*)</Solution>'

        def extract_first_match(matches : list[str]):
            
            return "".join(matches[-1]) if matches else None

        return DataFrameParser.transform_colums(df, column_list, answer_pattern, extract_first_match, new_col_suffix="_parsed")

    @staticmethod
    def evaluate_yes_no(df, column_list):
        """
        Evaluates the last occurrence of ###yes or ###no tags in the specified columns.

        Parameters:
        - df (pd.DataFrame): The input dataframe containing the columns to be parsed.
        - column_list (list): A list of column names to search and evaluate yes/no tags.


        Returns:
        - pd.DataFrame: The original dataframe with additional columns containing the evaluation results.
        """
       
        yes_no_pattern = r'###(yes|no)'

        def determine_last_decision(matches):
            if not matches:
                return None
            last_tag = matches[-1]
            return last_tag == 'yes' # otherwise it must be '###no'

        return DataFrameParser.transform_colums(df, column_list, yes_no_pattern, determine_last_decision, new_col_suffix="_decision")

    def split_csvs(df, columns):
        """
        Splits the strings at ', ' for specified columns in the DataFrame.
        If a value is `None`, it stays `None`.
        If a string is empty, it becomes an empty array.
        """
        def split_entry(val):
            if val is None:
                return None
            elif isinstance(val, str):
                return[entry.strip() for entry in  val.split(',')] if val else []
            else:
                print("VALUE", val, type(val))
                raise ValueError("Expected None or string, got something else.")

        for column in columns:
            print("COLUMN", column)
            extracted_series = df[column].astype(str)
            df[column + "_split"] = extracted_series.apply(split_entry)

        return df
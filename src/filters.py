from abc import ABC, abstractmethod
import pandas as pd


class Filter(ABC):


    def __init__(self, config_loader ):
        pass
        

    @abstractmethod
    def apply(self,input_df : pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass


class ValidFilter(Filter):
    # filter that returns only invalid rows
    validation_function : callable = None
    output_column : str = None
    id_column : str = None
    output_df : pd.DataFrame = None
    

    def __init__(self, config_loader ):
        super().__init__(config_loader= config_loader)
        self.validation_function = config_loader.prompt_config["validation_function"]
        self.output_column = config_loader.model_config["output_column"]
        self.id_column = config_loader.dataset_config["id_column"]
        self.output_df = pd.read_json(config_loader.get_model_output_dataset_path(config_loader.dataset_filename, config_loader.prompt_name), orient='records', lines=True, dtype=False, convert_axes=False)
    
    def apply(self, df : pd.DataFrame, **kwargs)-> pd.DataFrame:
        invalid_rows = []

        # Iterate through each row in output_df
        for _, row in self.output_df.iterrows():
            # Get the value from the output_column
            value_to_validate = row[self.output_column]

            # Apply the validation function
            if not self.validation_function(value_to_validate):
                # Find the corresponding row in input_df using the id_column
                matching_row = df[df[self.id_column] == row[self.id_column]]

                # Append the row from input_df to the invalid_rows list
                if not matching_row.empty:
                    invalid_rows.append(matching_row.iloc[0])
                else:
                    print("ERROR could not find matching row in dataframe for id",row[self.id_column] )

        # Convert the list of invalid rows to a DataFrame
        result_df = pd.DataFrame(invalid_rows, columns=df.columns)

        return result_df

       
class UnanswerableFilter(Filter):

    def __init__(self,config_loader ):
        super().__init__(config_loader= config_loader)
        self.label_column = config_loader.dataset_config["label_column"]

    
    def apply(self, df : pd.DataFrame, **kwargs) -> pd.DataFrame: 
        return df[df[self.label_column] != ""]

    


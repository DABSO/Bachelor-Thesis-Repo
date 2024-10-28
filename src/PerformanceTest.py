import pandas as pd
import numpy as np

class PerformanceTest():
    df: pd.DataFrame
    feature_columns: list[str]
    label_df: pd.DataFrame
    id_column: str

    def __init__(self, df: pd.DataFrame, label_df: pd.DataFrame, id_column: str) -> None:
        # Ensure both dataframes have id_column column
        if id_column not in df.columns or id_column not in label_df.columns:
            raise ValueError("Both dataframes must have an id_column column")
        self.id_column = id_column
        # Filter df to only include rows with IDs present in label_df
        self.df = df[df[id_column].isin(label_df[id_column])]

        # Filter label_df to only include rows with IDs present in df
        self.label_df = label_df[label_df[id_column].isin(self.df[id_column])]

        # Sort both dataframes by id_column to ensure they're in the same order
        self.df = self.df.sort_values(id_column).reset_index(drop=True)
        self.label_df = self.label_df.sort_values(id_column).reset_index(drop=True)

        # Verify that the IDs match after sorting
        if not (self.df[id_column] == self.label_df[id_column]).all():
            raise ValueError("IDs in df and label_df do not match after filtering and sorting")

    

    def run(self):
        # evaluate majority vote
        majority_vote_metrics = self.evaluate_votes("majority_vote")


        # evaluate unanimous vote
        unanimous_vote_metrics = self.evaluate_votes("unanimous_vote")


        # evaluate percentage vote
        percentage_vote_metrics = self.evaluate_percentage_vote()
        return majority_vote_metrics, unanimous_vote_metrics, percentage_vote_metrics
    

    def evaluate_votes(self, vote_column):
        labels = list(map(lambda l: l == False, self.label_df["matches_original"].to_list())) # False means the label is mislabeled -> invert to True means the label is incorrect
        predictions = list(map(PerformanceTest.invert_predction,self.df[vote_column].to_list())) # False means the prediction is mislabeled -> invert to True means the prediction is incorrect
        TP, FP, TN, FN, none_as_fn, none_as_fp = PerformanceTest.compute_confusion_values(predictions, labels)
        accuracy, precision, recall, f1_score, none_prediction_percentage = PerformanceTest.calculate_metrics_with_none(TP, FP, TN, FN, none_as_fn, none_as_fp)
        return {
            'column': vote_column,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'no_prediction' : none_prediction_percentage
        }

        
        
    def evaluate_auc_pr(self, column):
        labels = list(map(lambda l: 1 if l == True else 0, self.label_df["matches_original"].to_list())) 
        predictions = self.df[column].to_list()


        def calculate_precision_recall(y_true, y_scores):
            # Combine scores and labels, then sort by score in ascending order
            combined = sorted(zip(y_scores, y_true), key=lambda x: x[0])
            
            total_positives = sum([1 for label in y_true if label == 0])
            precisions, recalls = [], []
            true_positives = 0
            
            for i, (score, label) in enumerate(combined, 1):
                if label == 0:  
                    true_positives += 1

                
                precision = true_positives / i if i > 0 else 0
                recall = true_positives / total_positives if total_positives > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)

            return precisions, recalls
        

        precisions, recalls = calculate_precision_recall(labels, predictions)
        auc_score = np.trapz(precisions, recalls)
        return auc_score

        
    


   

        
    def evaluate_percentage_vote(self):
        treshold_overall_metrics_df = pd.DataFrame(columns=['column', 'threshold','accuracy' ,'precision', 'recall', 'f1_score'])

        percentage_columns = ["percentage_vote"]

        for col in percentage_columns:
            for threshold in np.arange(0.0, 1.05, 0.05):
                predictions = list(map(lambda p: p is not None and p <= threshold , self.df[col].to_list()))
                labels = list(map(lambda l: l == False, self.label_df["matches_original"].to_list()))

                TP, FP, TN, FN, none_as_fn, none_as_fp = PerformanceTest.compute_confusion_values(predictions, labels)
                

                accuracy, precision, recall, f1_score, none_prediction_percentage = PerformanceTest.calculate_metrics_with_none(TP, FP, TN, FN, none_as_fn, none_as_fp)
                
                treshold_overall_metrics_df = treshold_overall_metrics_df._append({
                    'column': col,
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                }, ignore_index=True)
        return treshold_overall_metrics_df

    @staticmethod
    def compute_confusion_values(predictions, labels):
        TP = FP = TN = FN = none_as_fn = none_as_fp = 0

        for pred, actual in zip(predictions, labels):
            
            if pred is None:
                if actual:  # When the label is True, but the prediction is None, it's a missed positive case
                    none_as_fn += 1
                else:  # When the label is False, but the prediction is None, it's a missed negative case
                    none_as_fp += 1
            else:
                if actual and pred:
                    TP += 1
                elif not actual and not pred:
                    TN += 1
                elif not actual and pred:
                    FP += 1
                elif actual and not pred:
                    FN += 1

        return TP, FP, TN, FN, none_as_fn, none_as_fp

    	
    @staticmethod
    def calculate_metrics_with_none(TP, FP, TN, FN, none_as_fn, none_as_fp):
        TP += 0  # None Werte beeinflussen TP nicht direkt
        FN += none_as_fn  # None Werte werden als FN gezählt
        TN += 0  # None Werte beeinflussen TN nicht direkt

        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        no_prediction_percentage = (none_as_fp + none_as_fn) / total

        return accuracy, precision, recall, f1_score, no_prediction_percentage


    @staticmethod
    def invert_predction(pred):
        if pd.isna(pred):
            return None
        elif pred is None:
            return None
        elif pred == True: 
            return False
        elif pred == False:
            return True



    

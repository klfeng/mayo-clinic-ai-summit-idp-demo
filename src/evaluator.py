import pandas as pd
import boto3
from fuzzywuzzy import fuzz
import io
import json

class Evaluator:
    def __init__(self):
        self.bedrock_client = boto3.client("bedrock-runtime")
        self.total_count = 0


    def create_comparison_df(self, ground_truth_path: str, results_path: str) -> pd.DataFrame:
        ground_truth_df = pd.read_csv(ground_truth_path)
        llm_results_df = pd.read_csv(results_path)
        comparison_df = llm_results_df.merge(ground_truth_df, on="field_name", how="left")
        return comparison_df
    
    
    def get_exact_match(self, df: pd.DataFrame) -> pd.DataFrame:
        if "llm_extraction" in df.columns:
            df['exact_match'] = df['field_value'] == df['llm_extraction']
        else:
            df['exact_match'] = df['field_value'] == df['bda_value']
        return df
    

    def get_fuzzy_match(self, df: pd.DataFrame) -> pd.DataFrame:
        def fuzzy_match(a, b):
            return fuzz.ratio(str(a).lower(), str(b).lower()) >= 80
        df['fuzzy_match'] = False  
        if "llm_extraction" in df.columns:
            df.loc[~df['exact_match'], 'fuzzy_match'] = df[~df['exact_match']].apply(
                    lambda row: fuzzy_match(row['field_value'], row['llm_extraction']), 
                    axis=1
                )
        else:
            df.loc[~df['exact_match'], 'fuzzy_match'] = df[~df['exact_match']].apply(
                    lambda row: fuzzy_match(row['field_value'], row['bda_value']), 
                    axis=1
                )
        return df
    
    def get_llm_match(self, df: pd.DataFrame, fuzzy_match: bool = False) -> pd.DataFrame:
        def llm_similarity_check(field_value, llm_extraction):
            system_prompt = f"""
            Your task is to evaluate whether two texts are similar or not. 
            You must only return True or False. Do no include anything else in your response.
            """
            user_prompt = f"""
            Compare the following two texts and determine if they are semantically similar:
            Text 1: {field_value}
            Text 2: {llm_extraction}
            Respond with only 'True' if they are semantically similar, or 'False' if they are not.
            Do not provide any explanation or additional text in your response.
            """
            body = json.dumps(
                {
                    "max_tokens": 5,
                    "system": system_prompt,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "anthropic_version": "bedrock-2023-05-31",
                }
            )
            response = self.bedrock_client.invoke_model(
                modelId="us.anthropic.claude-3-5-haiku-20241022-v1:0",
                body=body
            )
            response_body = json.loads(response['body'].read())
            llm_response = response_body["content"][0]["text"]
            return llm_response == 'True'
        
        df['llm_match'] = False
        if fuzzy_match:
            mask = (~df['exact_match']) & (~df['fuzzy_match'])
        else:
            mask = ~df['exact_match']

        if "llm_extraction" in df.columns:
            df.loc[mask, 'llm_match'] = df[mask].apply(
                lambda row: llm_similarity_check(row['field_value'], row['llm_extraction']),
                axis=1
            )  
        else:
            df.loc[mask, 'llm_match'] = df[mask].apply(
                lambda row: llm_similarity_check(row['field_value'], row['bda_value']),
                axis=1
            ) 
        return df
            


    def calculate_accuracy(self, df: pd.DataFrame, match_type: str = "EXACT") -> pd.DataFrame:        
        self.total_count = len(df)
        comparison_df = df.copy()
        if match_type == "EXACT":
            exact_match_df = self.get_exact_match(df=comparison_df)
            exact_match_accuracy = (exact_match_df['exact_match'].sum() / self.total_count) * 100
            print(f"Exact match accuracy: {exact_match_accuracy:.2f}%")
            return exact_match_df
        elif match_type == "FUZZY":
            exact_match_df = self.get_exact_match(df=comparison_df)
            fuzzy_match_df = self.get_fuzzy_match(df=exact_match_df)
            fuzzy_match_accuracy = ((fuzzy_match_df["fuzzy_match"].sum() + fuzzy_match_df["exact_match"].sum()) / self.total_count) * 100
            print(f"Fuzzy match accuracy: {fuzzy_match_accuracy:.2f}%")
            return fuzzy_match_df
        elif match_type == "LLM":
            exact_match_df = self.get_exact_match(df=comparison_df)
            llm_match_df = self.get_llm_match(df=exact_match_df)
            llm_match_accuracy = ((llm_match_df["llm_match"].sum() + llm_match_df["exact_match"].sum()) / self.total_count) * 100
            print(f"LLM match accuracy: {llm_match_accuracy:.2f}%")
            return llm_match_df
        elif match_type == "FUZZY_AND_LLM":
            exact_match_df = self.get_exact_match(df=comparison_df)
            fuzzy_match_df = self.get_fuzzy_match(df=exact_match_df)
            llm_fuzzy_match_df = self.get_llm_match(df=fuzzy_match_df, fuzzy_match=True)
            llm_fuzzy_match_accuracy = ((llm_fuzzy_match_df["llm_match"].sum() + llm_fuzzy_match_df["fuzzy_match"].sum() + llm_fuzzy_match_df["exact_match"].sum()) / self.total_count) * 100
            print(f"LLM and Fuzzy match accuracy: {llm_fuzzy_match_accuracy:.2f}%")
            return llm_fuzzy_match_df
        else:
            print(f"Match type must be one of: EXACT, FUZZY, LLM, FUZZY_AND_LLM")     

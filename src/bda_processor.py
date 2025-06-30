import boto3
import json
import pandas as pd
import os

class BDAProcessor:
    def __init__(self):
        self.s3_client = boto3.client("s3")
        self.bda_client = boto3.client("bedrock-data-automation")
        self.bda_runtime_client = boto3.client("bedrock-data-automation-runtime")
        self.bucket_name = "mayo-clinic-ai-summit-demo-files"
        self.sts_client = boto3.client('sts')
        self.account_id = self.sts_client.get_caller_identity()['Account']
        self.results_output_path = "output/results/bda_results/"
        self.file_name = None


    def create_blueprint(self, blueprint_name: str, blueprint_schema: dict) -> str:
        response = self.bda_client.create_blueprint(
            blueprintName=blueprint_name,
            type="DOCUMENT",
            blueprintStage="LIVE",
            schema=json.dumps(blueprint_schema)
        )
        return response["blueprint"]["blueprintArn"]
    

    def create_bda_project(self, project_name: str) -> str:
        standard_output_config =  {
            "document": {
                "extraction": {
                "granularity": {"types": ["DOCUMENT","PAGE", "ELEMENT","LINE","WORD"]},
                "boundingBox": {"state": "ENABLED"}
                },
                "generativeField": {"state": "ENABLED"},
                "outputFormat": {
                "textFormat": {"types": ["PLAIN_TEXT", "MARKDOWN", "HTML", "CSV"]},
                "additionalFileFormat": {"state": "ENABLED"}
                }
            }
        }
        response = self.bda_client.create_data_automation_project(
            projectName=project_name,
            projectDescription="BDA project for Mayo Clinic AI Summit demo",
            projectStage='LIVE',
            standardOutputConfiguration=standard_output_config
        )
        return response["projectArn"]
    

    def upload_to_s3(self, file_path: str) -> None:
        s3_response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix="input_files")
        current_files = []
        for obj in s3_response.get("Contents", []):
            current_files.append(obj["Key"])
        if file_path not in current_files:
            self.s3_client.upload_file(
                Filename=file_path,
                Bucket=self.bucket_name,
                Key=file_path
            )
    

    def start_data_automation(self, file_path: str, blueprint_arn: str) -> str:
        self.file_name = os.path.basename(file_path).split(".")[0]
        self.upload_to_s3(file_path=file_path)
        response = self.bda_runtime_client.invoke_data_automation_async(
            inputConfiguration={
                's3Uri': f"s3://{self.bucket_name}/{file_path}"
            },
            outputConfiguration={
                's3Uri': f"s3://{self.bucket_name}/bda_results"
            },
            blueprints=[
                {
                    'blueprintArn': blueprint_arn
                }
            ],
            dataAutomationProfileArn=f"arn:aws:bedrock:us-east-1:{self.account_id}:data-automation-profile/us.data-automation-v1",
        )
        return response["invocationArn"].split("/")[1]
    

    def get_data_automation_results(self, job_id: str) -> pd.DataFrame:
        response = self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key=f"bda_results/{job_id}/0/custom_output/0/result.json"
        )
        body = json.loads(response['Body'].read().decode('utf-8'))
        results = body["inference_result"]
        df = pd.DataFrame(results.items(), columns=['field_name', 'bda_value'])
        df.to_csv(f"{self.results_output_path}{self.file_name}_bda_results.csv", index=False)
        return df

    

    
    


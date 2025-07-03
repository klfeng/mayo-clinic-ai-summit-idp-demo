import boto3
import json
import pandas as pd
import os
import sagemaker

class BDAProcessor:
    def __init__(self):
        self.session = sagemaker.Session()
        self.default_bucket = self.session.default_bucket()
        self.s3_client = boto3.client("s3")
        self.bda_client = boto3.client("bedrock-data-automation")
        self.bda_runtime_client = boto3.client("bedrock-data-automation-runtime")
        self.sts_client = boto3.client('sts')
        self.account_id = self.sts_client.get_caller_identity()['Account']
        self.results_output_path = "output/"
        self.file_name = None


    def create_blueprint(self, blueprint_name: str, blueprint_schema: dict) -> str:
        response = self.bda_client.create_blueprint(
            blueprintName=blueprint_name,
            type="DOCUMENT",
            blueprintStage="LIVE",
            schema=json.dumps(blueprint_schema)
        )
        blueprint_arn = response["blueprint"]["blueprintArn"]
        print("Successfully created blueprint")
        return blueprint_arn
    
    def update_blueprint(self, blueprint_arn: str, blueprint_schema: dict) -> str:
        response = self.bda_client.update_blueprint(
            blueprintArn=blueprint_arn,
            schema=json.dumps(blueprint_schema)
        )
        print("Successfully updated blueprint")
    

    def upload_to_s3(self, file_path: str) -> None:
        s3_response = self.s3_client.list_objects_v2(Bucket=self.default_bucket, Prefix="input_files")
        current_files = []
        for obj in s3_response.get("Contents", []):
            current_files.append(obj["Key"])
        if file_path not in current_files:
            self.s3_client.upload_file(
                Filename=file_path,
                Bucket=self.default_bucket,
                Key=file_path
            )
    

    def start_data_automation(self, file_path: str, blueprint_arn: str) -> str:
        self.file_name = os.path.basename(file_path).split(".")[0]
        self.upload_to_s3(file_path=file_path)
        response = self.bda_runtime_client.invoke_data_automation_async(
            inputConfiguration={
                's3Uri': f"s3://{self.default_bucket}/{file_path}"
            },
            outputConfiguration={
                's3Uri': f"s3://{self.default_bucket}/bda_results"
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
        try:
            response = self.s3_client.get_object(
                Bucket=self.default_bucket,
                Key=f"bda_results/{job_id}/0/custom_output/0/result.json"
            )
            body = json.loads(response['Body'].read().decode('utf-8'))
            results = body["inference_result"]
            df = pd.DataFrame(results.items(), columns=['field_name', 'bda_value'])
            df.to_csv(f"{self.results_output_path}processed_{self.file_name}.csv", index=False)
            return df
        except Exception as e:
            print("Document extraction is still in progress. Please try again later.")

    

    
    


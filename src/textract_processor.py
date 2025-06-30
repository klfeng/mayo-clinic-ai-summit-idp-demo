import boto3
import json
import time
import os
from textractor.parsers.response_parser import parse
from textractor.data.markdown_linearization_config import MarkdownLinearizationConfig

class TextractProcessor:
    def __init__(self):
        self.textract_client = boto3.client("textract")
        self.s3_client = boto3.client("s3")
        self.bucket_name = "mayo-clinic-ai-summit-demo-files"
        self.textract_output_path = "output/textract_output/"
        self.markdown_output_path = "output/markdown_output/"
        self.file_name = None

    def upload_to_s3(self, file_path: str):
        self.s3_client.upload_file(
            Filename=file_path,
            Bucket=self.bucket_name,
            Key=file_path
        )

    def start_textract(self, file_path: str) -> str:
        try:
            self.upload_to_s3(file_path=file_path)
            self.file_name = os.path.basename(file_path).split(".")[0]
            response = self.textract_client.start_document_analysis(
                DocumentLocation={
                    "S3Object": {
                        "Bucket": self.bucket_name,
                        "Name": file_path,
                    }
                },
                FeatureTypes=["FORMS", "TABLES", "LAYOUT"],
            )
            return response["JobId"]
        except Exception as e:
            raise RuntimeError(f"Error starting Textract: {str(e)}") from e 
    

    def get_textract_response(self, job_id: str) -> dict:
        def check_job_status(job_id: str):
            textract_response = self.textract_client.get_document_analysis(
                JobId=job_id, MaxResults=1000
            )
            status = textract_response["JobStatus"]
            if status == "SUCCEEDED":
                return True
            else:
                time.sleep(30)
                return check_job_status(job_id)
    
        try:
            check_job_status(job_id)
            textract_response = self.textract_client.get_document_analysis(
                JobId=job_id, MaxResults=1000
            )
            if "NextToken" in textract_response:
                not_finished = True
                next_token = textract_response["NextToken"]
            else:
                not_finished = False
            while not_finished:
                next_response = self.textract_client.get_document_analysis(
                    JobId=job_id, MaxResults=1000, NextToken=next_token
                )
                textract_response["Blocks"].extend(next_response["Blocks"])
                if "NextToken" in next_response:
                    not_finished = True
                    next_token = next_response["NextToken"]
                else:
                    not_finished = False

            with open(f"{self.textract_output_path}{self.file_name}.json", "w") as f:
                json.dump(textract_response, f, indent=4)

            return textract_response
        except Exception as e:
            raise RuntimeError(f"Error getting Textract response: {str(e)}") from e 
    

    def parse_textract_response(self, textract_response: dict) -> str:
        try:
            document = parse(textract_response)
            markdown_content = document.to_markdown(
                MarkdownLinearizationConfig(
                    page_num_prefix="---Page---", hide_page_num_layout=False
                )
            )
            pages = markdown_content.split("---Page---")
            pages_w_num = [
                f"--- Page {index + 1}---\n{page}" for index, page in enumerate(pages)
            ]
            document_content = "".join(pages_w_num)

            with open(f"{self.markdown_output_path}{self.file_name}.md", "w", encoding="utf-8") as file:
                file.write(document_content)

            return document_content
        except Exception as e:
            raise RuntimeError(f"Error parsing Textract response: {str(e)}") from e 

        
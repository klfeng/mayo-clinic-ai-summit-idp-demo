import boto3
import pandas as pd
import json
import re
import os

class Extractor:
    def __init__(self):
        self.bedrock_client = boto3.client("bedrock-runtime")
        self.results_output_path = "output/results/llm_results/"
        self.model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        self.file_name = None


    def load_schema(self, schema_path: str) -> list:
        field_definitions = pd.read_csv(schema_path)
        field_definitions_json = json.loads(field_definitions.to_json(orient="records"))
        return field_definitions_json
    

    def load_document(self, document_path: str) -> str:
        with open(document_path, "r") as file:
            document_content = file.read()
        return document_content


    def create_schema(self, field: dict) -> str:
        field_name = field["field_name"]
        field_definition = field["field_definition"]
        field_instructions = field["field_instructions"]
        field_desc = [f"Field: {field_name}"]
        if field_definition:
            field_desc.append(f"Definition: {field_definition}")
        if field_instructions:
            field_desc.append(f"Instructions: {field_instructions}")
        return " ".join(field_desc)


    def _remove_double_quotes(self, json_str: str) -> dict:
        lines = json_str.split("\n")
        new_lines = []
        total_lines = len(lines)
        for index, line in enumerate(lines):
            new_line = line
            if ":" in line:
                first_colon_index = line.find(":")
                after_colon = line[first_colon_index + 1 :]
                first_quote_index = after_colon.find('"')
                last_quote_index = after_colon.rfind('"')
                if first_quote_index > -1 and last_quote_index > -1:
                    middle_part = after_colon[first_quote_index + 1 : last_quote_index]
                    middle_part = middle_part.replace('"', "")
                    new_line = line[: first_colon_index + 1] + f'"{middle_part}"'
                    if index + 1 < total_lines - 1:
                        if lines[index + 1].strip() != "}":
                            new_line += ","
            new_lines.append(new_line)
        new_json_string = "\n".join(new_lines)
        return json.loads(new_json_string)


    def _claude_json_cleanup(self, json_str: str) -> dict:
        system_prompt_json = """
        Your task is to parse incorrect or corrupt JSON string.
        You must only return a valid JSON string as your response. Do no include anything else in your response.
        """
        user_prompt_json = f"""
        Given the invalid JSON string, return a valid JSON string.
        There might be missing quotes or double quotes or weird indents that is making it an invalid JSON.
        I need to be able to parse this JSON string without any errors. Remember to only include the JSON string in your response.
        Only return a JSON string as your output.
        {json_str}
        """
        body_json = json.dumps(
            {
                "max_tokens": 200000,
                "system": system_prompt_json,
                "temperature": 0,
                "messages": [{"role": "user", "content": user_prompt_json}],
                "anthropic_version": "bedrock-2023-05-31",
            }
        )
        updated_response = self.bedrock_client.invoke_model(
            body=body_json, 
            modelId=self.model_id
        )
        updated_response_body = json.loads(updated_response.get("body").read())
        updated_response_content = updated_response_body["content"][0]["text"]
        try:
            updated_json_content = json.loads(updated_response_content)
            return updated_json_content
        except Exception as e:
            updated_json_string_match = re.search(
                r"{.*}", updated_response_content, re.DOTALL
            )
            if updated_json_string_match:
                updated_json_string = updated_json_string_match.group(0)
                updated_json_string = updated_json_string.replace("'", '"')
                updated_json_output = json.loads(updated_json_string)
                return updated_json_output
        

    def invoke_model(self, system_prompt: str, extraction_prompt: str) -> str:
        body = json.dumps(
            {
                "max_tokens": 200000,
                "system": system_prompt,
                "messages": [{"role": "user", "content": extraction_prompt}],
                "anthropic_version": "bedrock-2023-05-31",
            }
        )
        response = self.bedrock_client.invoke_model(
            body=body,
            modelId=self.model_id
        )
        response_body = json.loads(response.get("body").read())
        response_content = response_body["content"][0]["text"]
        try:
            json_content = json.loads(response_content)
            return json_content
        except json.JSONDecodeError as e:
            json_string_match = re.search(r"{.*}", response_content, re.DOTALL)
            if json_string_match:
                json_string = json_string_match.group(0)
                json_string = json_string.replace("'", '"')
                try:
                    return json.loads(json_string)
                except json.JSONDecodeError as e:
                    try:
                        field_json = self._remove_double_quotes(json_string)
                        return field_json
                    except json.JSONDecodeError as e:
                        field_json = self._claude_json_cleanup(json_string)
                        return field_json
            else:
                return {}
            

    def save_results(self, results: dict) -> None:
        csv_data = []
        for field_name, field_data in results.items():
            csv_data.append({
                'field_name': field_name,
                'llm_extraction': field_data.get('variable_field_name', ''),
                'reasoning': field_data.get('reasoning', ''),
                'page_citation': field_data.get('page_citation', '')
            })
        df = pd.DataFrame(csv_data)
        df.to_csv(f"{self.results_output_path}{self.file_name}_llm_results.csv", index=False)


    def extract_metadata(self, document_path: str, system_prompt: str, extraction_prompt: str, field_definitions_path: str) -> dict:
        self.file_name = os.path.basename(document_path).split(".")[0]
        document = self.load_document(document_path=document_path)
        fields = self.load_schema(schema_path=field_definitions_path)
        results = {}
        for field in fields:
            print("Processing field:", field["field_name"])
            field_schema = self.create_schema(field=field)
            field_prompt = extraction_prompt.format(
                schema=field_schema,
                document=document
            )
            response = self.invoke_model(
                system_prompt=system_prompt,
                extraction_prompt=field_prompt
            )
            results.update(response)  
        self.save_results(results=results)
        
import os
import json
import pandas as pd
import re
from dotenv import load_dotenv
import openai
import boto3
import together

# Load API keys from .env file
load_dotenv("api_keys.env")

# Retrieve API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# AWS Bedrock Client (only initialize if keys exist)
bedrock = None
if AWS_ACCESS_KEY and AWS_SECRET_KEY:
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

# Together AI Setup
if TOGETHER_API_KEY:
    together.api_key = TOGETHER_API_KEY

# Load flight dataset
FLIGHT_DATA_PATH = "C:/Users/guna laakshmi/Downloads/submission/problem1/flights_dataset.csv"

if not os.path.exists(FLIGHT_DATA_PATH):
    raise FileNotFoundError(f"Flight data file not found: {FLIGHT_DATA_PATH}")

flights_df = pd.read_csv(FLIGHT_DATA_PATH)

def get_flight_info(flight_number: str) -> dict:
    """
    Fetch flight details from the dataset.
    Returns a dictionary with flight details or an error message.
    """
    flight = flights_df[flights_df["flight_number"] == flight_number]
    
    if flight.empty:
        return {"error": f"Flight {flight_number} not found in database."}
    
    return flight.iloc[0].to_dict()

def info_agent_request(flight_number: str) -> str:
    """
    Calls get_flight_info and returns the data as a structured JSON string.
    """
    flight_info = get_flight_info(flight_number)
    return json.dumps(flight_info, indent=2)

def extract_flight_number(query: str) -> str:
    """
    Extracts the flight number from the user query using regex.
    Matches common flight formats (AI123, BA456, etc.).
    """
    match = re.search(r"\b[A-Z]{2}\d{2,4}\b", query)  # More flexible regex
    return match.group() if match else None

def call_openai(prompt: str) -> str:
    """ Query OpenAI API with error handling """
    if not OPENAI_API_KEY:
        return "OpenAI API key is missing."

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use gpt-3.5-turbo instead of gpt-4
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        return f"OpenAI API error: {str(e)}"

def call_aws_bedrock(prompt: str) -> str:
    """ Query AWS Bedrock API with error handling """
    if not bedrock:
        return "AWS Bedrock API key is missing."

    try:
        response = bedrock.invoke_model(
            body=json.dumps({"prompt": prompt, "max_tokens": 100}),
            modelId="anthropic.claude-v2"
        )
        return response["body"].read().decode("utf-8")
    except Exception as e:
        return f"AWS Bedrock API error: {str(e)}"

def call_together_ai(prompt: str) -> str:
    """ Query Together AI API (free alternative) """
    if not TOGETHER_API_KEY:
        return "Together AI API key is missing."

    try:
        response = together.ChatCompletion.create(
            model="together/gpt-neoxt-20b",  # Free model
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Together AI API error: {str(e)}"

def qa_agent_respond(user_query: str, provider: str = "openai") -> str:
    """
    Extracts flight number, calls info_agent_request, and formats the response using AI.
    """
    flight_number = extract_flight_number(user_query)
    
    if not flight_number:
        return json.dumps({"answer": "No valid flight number found in query."})

    flight_data_json = info_agent_request(flight_number)
    flight_data = json.loads(flight_data_json)

    if "error" in flight_data:
        return json.dumps({"answer": flight_data["error"]})

    prompt = f"Flight {flight_data['flight_number']} departs at {flight_data['departure_time']} " \
             f"to {flight_data['destination']}. Current status: {flight_data['status']}."

    # Choose AI provider with failover
    if provider == "openai" and OPENAI_API_KEY:
        ai_response = call_openai(prompt)
    elif provider == "aws" and AWS_ACCESS_KEY:
        ai_response = call_aws_bedrock(prompt)
    elif provider == "together" and TOGETHER_API_KEY:
        ai_response = call_together_ai(prompt)
    else:
        # Default to Together AI if other APIs fail
        ai_response = call_together_ai(prompt) if TOGETHER_API_KEY else "No available AI providers."

    response = {"answer": ai_response}
    return json.dumps(response, indent=2)

# Testing the functions
if __name__ == "__main__":
    print(qa_agent_respond("When does Flight AI123 depart?", provider="openai"))  # OpenAI
    print(qa_agent_respond("What is the status of Flight AI999?", provider="aws"))  # AWS Bedrock
    print(qa_agent_respond("Tell me about Flight AI456?", provider="together"))  # Together AI

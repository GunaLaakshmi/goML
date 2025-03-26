# Flight Info Query System

## Overview
This project provides a flight information query system that extracts flight details from a dataset and enhances responses using AI models such as OpenAI, AWS Bedrock, and Together AI.

## Features
- Extracts flight details from a CSV dataset.
- Supports multiple AI providers (OpenAI, AWS Bedrock, Together AI) for enhanced responses.
- Uses environment variables for API key management.
- Implements error handling and failover mechanisms.

## Prerequisites
Ensure you have the following installed:
- Python 3.x
- Required Python packages (install via `pip install -r requirements.txt`)
- API keys for OpenAI, AWS Bedrock, or Together AI
- Dataset file: `flights_dataset.csv`

## Setup

### 1. Install Dependencies
```sh
pip install pandas openai boto3 together dotenv
```

### 2. Create an `.env` File
Create a file named `api_keys.env` in the project directory and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY=your_aws_access_key
AWS_SECRET_KEY=your_aws_secret_key
TOGETHER_API_KEY=your_together_ai_key
```

### 3. Ensure the Flight Dataset is Available
Place the dataset in the correct directory:
```
C:/Users/guna laakshmi/Downloads/submission/problem1/flights_dataset.csv
```

## Usage
Run the script and query flight information using AI providers:
```sh
python flight_query.py
```

### Example Queries
```python
qa_agent_respond("When does Flight AI123 depart?", provider="openai")
qa_agent_respond("What is the status of Flight AI999?", provider="aws")
qa_agent_respond("Tell me about Flight AI456?", provider="together")
```

## API Providers
- **OpenAI** (`GPT-3.5 Turbo`)
- **AWS Bedrock** (`Claude v2`)
- **Together AI** (`GPT-NeoXT-20B`)

## Error Handling
- If an API key is missing, the system will return an appropriate error message.
- If a flight number is not found in the dataset, an error message will be returned.
- If the primary AI provider fails, the system falls back to Together AI.

## License
This project is licensed under the MIT License.



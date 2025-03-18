# Domain-Specific Semantic-Rich Software Knowledge Graph Construction through Human-LLM Team Working

## Project Overview
This project focuses on constructing a domain-specific semantic-rich software knowledge graph through collaborative efforts between humans and Large Language Models (LLMs). The project involves processing data, generating semantic representations, and building a knowledge graph that can be utilized for various software engineering tasks.

## Prerequisites

### Python Version
Ensure you have Python 3.12.2 installed on your system. You can check your Python version by running:
```bash
python --version
Installation
To install the required dependencies, run the following command:

bash
pip install -r requirements.txt
Configuration
YAML File Configuration
You need to configure the config.yaml file with the following fields:

yaml
OpenAI_API_Base: ""  # Base URL for the OpenAI API
API_key_list:       # List of API keys for authentication
  - 
chunk_size:         # Size of data chunks for processing
OpenAI_API_Base: Specify the base URL for the OpenAI API.

API_key_list: Provide a list of API keys that will be used for authentication.

chunk_size: Define the size of data chunks that will be processed at a time.

Running the Code
Execution
To run the project, navigate to the Code directory and execute the main.py script located in the Code/new_code/ folder. You can do this by right-clicking on the main.py file and selecting "Run" or by using the following command in your terminal:

bash
python Code/new_code/main.py
Data and Code Structure
Data: The data required for the project should be placed in the data directory.

Code: The main script to run is located in Code/new_code/main.py.

Additional Notes
Ensure that the config.yaml file is correctly configured before running the script.

The project relies on the OpenAI API, so make sure you have valid API keys and the correct API base URL.

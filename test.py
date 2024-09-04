import google.generativeai as genai
from hashhop.generate import MultiHopEval
from tqdm import tqdm
from collections import defaultdict
import os
import random
import time
import argparse
import csv

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Constants
MODEL_NAME = "gemini-1.5-flash-exp-0827"  # Adjust if needed for Gemini Flash
API_KEY = os.environ.get("GOOGLE_API_KEY")
SCOPES = ['https://www.googleapis.com/auth/generative-language.retriever']

SLEEP_INTERVAL = 30 

safe = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

class HashIndex:
    def __init__(self, hash_assignments):
        self.index = self.build_index(hash_assignments)

    def build_index(self, hash_assignments):
        index = defaultdict(list)
        for line in hash_assignments.split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip("'")
                index[key].append(value)
        return index

    def get_chain(self, start_hash, max_hops=2):
        chain = [start_hash]
        current = start_hash
        for _ in range(max_hops):
            if current in self.index:
                next_value = self.index[current][0]
                chain.append(next_value)
                current = next_value
            else:
                break
        return chain

def retrieve_relevant_chains(hash_assignments, queries, max_hops=2):
    index = HashIndex(hash_assignments)
    relevant_assignments = set()
    
    for query in queries:
        chain = index.get_chain(query, max_hops)
        for i in range(len(chain) - 1):
            relevant_assignments.add(f"{chain[i]} = {chain[i+1]}")
    
    return "\n".join(relevant_assignments)

def generate_prompt(hash_assignments):
    return f"""You are a HashHop solver. Your task is to navigate through a chain of hash assignments and find the final quoted string value for a given starting hash variable. All problems involve exactly 2 hops to reach the final string.

Rules:
1. Each hash assignment follows the format: H1 = H2, H2 = 'final_value', etc.
2. Hash variables are not quoted (e.g., H1, H2).
3. The final value in the chain is always a quoted string.
4. You will be given a starting hash variable as a query.
5. Your response should ONLY be the final quoted string value, without the quotes.
6. Do not provide any explanation or additional text in your response.

Here are eight examples with Chain of Thought (CoT) reasoning:

Example 1:
Hash assignments:
H1 = H2
H2 = H3
H3 = 'apple'
H4 = H5
H5 = 'banana'

Query: H1
Thought process:
1. Start with H1
2. H1 points to H2 (first hop)
3. H2 points to H3 (second hop)
4. H3 contains the string 'apple'
Response: apple

Example 2:
Hash assignments:
H1 = H2
H2 = 'cherry'
H3 = H4
H4 = 'date'

Query: H3
Thought process:
1. Start with H3
2. H3 points to H4 (first hop)
3. H4 contains the string 'date' (second hop)
Response: date

Example 3:
Hash assignments:
H1 = H2
H2 = H3
H3 = 'elderberry'
H4 = H5
H5 = 'fig'

Query: H4
Thought process:
1. Start with H4
2. H4 points to H5 (first hop)
3. H5 contains the string 'fig' (second hop)
Response: fig

Example 4:
Hash assignments:
H1 = H2
H2 = 'grape'
H3 = H4
H4 = H5
H5 = 'honeydew'

Query: H3
Thought process:
1. Start with H3
2. H3 points to H4 (first hop)
3. H4 points to H5 (second hop)
4. H5 contains the string 'honeydew'
Response: honeydew

Example 5:
Hash assignments:
H1 = H2
H2 = H3
H3 = 'imbe'
H4 = H5
H5 = 'jackfruit'

Query: H1
Thought process:
1. Start with H1
2. H1 points to H2 (first hop)
3. H2 points to H3 (second hop)
4. H3 contains the string 'imbe'
Response: imbe

Example 6:
Hash assignments:
H1 = H2
H2 = 'kiwi'
H3 = H4
H4 = 'lemon'
H5 = H6
H6 = 'mango'

Query: H5
Thought process:
1. Start with H5
2. H5 points to H6 (first hop)
3. H6 contains the string 'mango' (second hop)
Response: mango

Example 7:
Hash assignments:
H1 = H2
H2 = H3
H3 = 'nectarine'
H4 = H5
H5 = 'orange'

Query: H4
Thought process:
1. Start with H4
2. H4 points to H5 (first hop)
3. H5 contains the string 'orange' (second hop)
Response: orange

Example 8:
Hash assignments:
H1 = H2
H2 = 'papaya'
H3 = H4
H4 = H5
H5 = 'quince'

Query: H3
Thought process:
1. Start with H3
2. H3 points to H4 (first hop)
3. H4 points to H5 (second hop)
4. H5 contains the string 'quince'
Response: quince

Now, solve the following HashHop problem:

Hash assignments:
{hash_assignments}

Query: {{query}}

Remember:
1. Always follow exactly 2 hops to reach the final string.
2. Your final answer should be the quoted string value, without the quotes.
3. Provide only the final answer, no explanations or steps.
"""

def evaluate_model(hash_assignments, targets, model):
    correct = 0
    total = len(targets)
    
    prompt_template = generate_prompt(hash_assignments)
    
    for query, expected in tqdm(targets.items(), desc="Evaluating queries"):
        full_prompt = prompt_template.format(query=query)
        # print(full_prompt)
        try:
            response = model.generate_content(full_prompt)
            model_output = response.text.strip()
            # print(model_output)
            # print(expected)
            if model_output == expected:
                correct += 1
        except Exception as e:
            print(f"Error processing query {query}: {e}")

        time.sleep(SLEEP_INTERVAL)
    
    accuracy = (correct / total) * 100
    return accuracy

def generate_cot_prompt(hash_assignments, query):
    prompt = f"""You are a HashHop solver. Your task is to navigate through a chain of hash assignments and find the final quoted string value for a given starting hash variable. All problems involve exactly 2 hops to reach the final string.

Rules:
1. Each hash assignment follows the format: H1 = H2, H2 = 'final_value', etc.
2. Hash variables are not quoted (e.g., H1, H2).
3. The final value in the chain is always a quoted string.
4. You will be given a starting hash variable as a query.
5. Your response should include a "Thinking:" section with your step-by-step reasoning, and a "Response:" section with only the final quoted string value, without the quotes.
6. Do not provide any additional text or explanation in your response.

Hash assignments:
{hash_assignments}

Query: {query}

Remember to structure your answer with "Thinking:" and "Response:" sections.
"""
    return prompt

def generate_train_data(hash_assignments, completion, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['input', 'output'])
        
        for line in completion.split('\n')[5:]:
            if '=' in line:
                query, hops = line.split(' = ', 1)
                hops = hops.strip("'").split(' = ')
                expected_output = hops[-1]
                
                input_prompt = generate_cot_prompt(hash_assignments, query)
                
                output = f"""Thinking:
1. Start with {query}
2. {query} points to {hops[0]} (first hop)
3. {hops[0]} contains the string '{hops[1]}' (second hop)

Response:
{expected_output}"""
                
                csvwriter.writerow([input_prompt, output])

def evaluate_fine_tuned_model(hash_assignments, targets, model):
    correct = 0

    all_items = list(targets.items())
    selected_items = random.sample(all_items, min(100, len(all_items)))
    selected_targets = dict(selected_items)
    total = len(selected_targets)
    
    for query, expected in tqdm(selected_targets.items(), desc="Evaluating fine-tuned model"):
        # print(hash_assignments)
        # print(query)
        # print(expected)
        full_prompt = generate_cot_prompt(hash_assignments, query)
        
        try:
            response = model.generate_content(full_prompt)
            model_output = response.text.strip()
            
            # Extract the response from the output
            response_section = model_output.split("Response:")[-1].strip()
            
            # print(response_section)
            # print(expected)

            if response_section == expected:
                correct += 1
        except Exception as e:
            print(f"Error processing query {query}: {e}")

        time.sleep(SLEEP_INTERVAL)
    
    accuracy = (correct / total) * 100
    return accuracy

def load_creds():
    """Converts `client_secret.json` to a credential object.

    This function caches the generated tokens to minimize the use of the
    consent screen.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemini Flash model on HashHop dataset, generate training data, or evaluate fine-tuned model.")
    parser.add_argument("--generate-train-data", action="store_true", help="Generate training data CSV file")
    parser.add_argument("--eval", action="store_true", help="Evaluate fine-tuned model")
    parser.add_argument("--retrieval", action="store_true", help="Use retrieval strategy")
    args = parser.parse_args()

    datapoint = MultiHopEval.make_one(
        n_chars_problem=1000000,
        num_queries=1,
        hops=2,
        hash_pair_str_length=16,
        chain_of_thought=False,
    )

    if args.generate_train_data:
        print("Generating training data CSV file...")
        datapoint = MultiHopEval.make_one(
            n_chars_problem=10000,
            num_queries=100,
            hops=2,
            hash_pair_str_length=16,
            chain_of_thought=True,
        )
        # print(datapoint.prompt)
        # print(datapoint.completion)
        # exit(0)
        generate_train_data(datapoint.prompt, datapoint.completion, "train_data.csv")
        print("Training data generated and saved to train_data.csv")
    elif args.eval:
        print("Evaluating fine-tuned model on HashHop dataset...")
        creds = load_creds()
        genai.configure(credentials=creds)
        model = genai.GenerativeModel(MODEL_NAME, safety_settings=safe)
        accuracy = evaluate_fine_tuned_model(datapoint.prompt, datapoint.targets, model)
        print(f"Fine-tuned model accuracy: {accuracy:.2f}%")
    else:
        print("Evaluating Gemini Flash model on HashHop dataset...")
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(MODEL_NAME, safety_settings=safe)
        all_items = list(datapoint.targets.items())
        selected_items = random.sample(all_items, min(100, len(all_items)))
        selected_targets = dict(selected_items)
        
        if args.retrieval:
            print("Using retrieval strategy...")
            relevant_assignments = retrieve_relevant_chains(datapoint.prompt, selected_targets.keys())
            # print(relevant_assignments)
            # print(datapoint.prompt)
            # print(selected_targets)
            accuracy = evaluate_model(relevant_assignments, selected_targets, model)
        else:
            accuracy = evaluate_model(datapoint.prompt, selected_targets, model)
        
        print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
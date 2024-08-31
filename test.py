import google.generativeai as genai
from hashhop.generate import MultiHopEval
from tqdm import tqdm
import os
import random
import time

# Constants
MODEL_NAME = "gemini-1.5-flash-exp-0827"  # Adjust if needed for Gemini Flash
API_KEY = os.environ.get("GOOGLE_API_KEY")

SLEEP_INTERVAL = 30 
datapoint = MultiHopEval.make_one(
    n_chars_problem=1000000,
    num_queries=1,
    hops=2,
    hash_pair_str_length=16,
    chain_of_thought=False,
)

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

# Set up the Gemini model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME, safety_settings=safe)

# print(len(datapoint.prompt))
# print(datapoint.completion)
# print(len(datapoint.targets))

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

def evaluate_model(hash_assignments, targets):
    correct = 0
    total = len(targets)
    
    prompt_template = generate_prompt(hash_assignments)
    
    for query, expected in tqdm(targets.items(), desc="Evaluating queries"):
        full_prompt = prompt_template.format(query=query)
        
        try:
            response = model.generate_content(full_prompt)
            model_output = response.text.strip()
            
            if model_output == expected:
                correct += 1
        except Exception as e:
            print(f"Error processing query {query}: {e}")

        time.sleep(SLEEP_INTERVAL)
    
    accuracy = (correct / total) * 100
    return accuracy

def main():
    if not API_KEY:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

    print("Evaluating Gemini Flash model on HashHop dataset...")

    all_items = list(datapoint.targets.items())
    selected_items = random.sample(all_items, min(100, len(all_items)))
    selected_targets = dict(selected_items)

    accuracy = evaluate_model(datapoint.prompt, selected_targets)
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
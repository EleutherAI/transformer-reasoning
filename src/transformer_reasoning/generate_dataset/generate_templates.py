import os
import random
from openai import OpenAI
from typing import List, Tuple
import re
from transformer_reasoning.utils import get_project_root

def load_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def query_llm(prompt: str) -> List[str]:
    api_key = load_api_key(get_project_root() / "openrouter_api_key.txt")  # Adjust the path as needed

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key = api_key  # Adjust the path as needed
    )

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "YOUR_SITE_URL",  # Replace with your actual site URL
                "X-Title": "YOUR_APP_NAME",  # Replace with your actual app name
            },
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates sentence templates for biographies."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
            n=3,  # Generate 3 responses
        )
        
        # Extract the generated templates from the response
        templates = [choice.message.content.strip() for choice in completion.choices]
        
        return templates
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def verify_template(template: str, attributes: List[str], top_n: int = 10) -> List[str]:
    # Regex patterns to verify syntax
    attribute_patterns = [r'\{' + attribute + r'\}' for attribute in attributes]
    other_brackets_pattern = r'\[.*?\]|\(.*?\)'
    curly_brackets_pattern = r'\{.*?\}'
    single_end_period_pattern = r'^[^.]*\.$'
    underscore_pattern = r'__'
    # Filter templates that match all criteria
    return (all([len(re.findall(ap, template)) == 1 for ap in attribute_patterns])
        and len(re.findall(curly_brackets_pattern, template)) == len(attributes)
        and not re.search(other_brackets_pattern, template)
        and re.match(single_end_period_pattern, template)
        and not re.search(underscore_pattern, template))
    
def filter_templates(templates: List[str]) -> List[str]:
    templates = list(set(templates))
    return sorted(templates, key=lambda x: len(x))

def generate_diverse_prompts(prompt_function, num_prompts=50):
    prompt_parameters = {
        'writing_style': ['concise', 'conversational'],
        'tone': ['casual', 'humorous', 'serious'],
        'sophistication': ['highbrow', 'lowbrow'],
        'formality': ['formal', 'informal'],
        'genre': ['academic', 'journalistic', 'technical'],
        'figurative_language': ['literal', 'idiomatic'],
        'emotion': ['neutral', 'excited'],
        'vocabulary_level': ['basic', 'intermediate', 'advanced'],
    }
    all_details = {'birth date': 'subject\'s date of birth',
               'birth city': 'subject\'s city of birth',
               'university': 'university the subject attended',
               'employer': 'subject\'s current employer'}
    
    prompts = []
    attributes = []
    for _ in range(num_prompts):
        chosen_params = random.sample(list(prompt_parameters.keys()), 3)
        chosen_detail = random.choice(list(all_details.keys()))
        order = random.choice([0, 1])
        ordered_details = ['name', chosen_detail]
        prompt = "Generate a brief sentence template that is "
        prompt += ", ".join([f"{random.choice(prompt_parameters[k])}" for k in chosen_params[:-1]])
        prompt += f" and {random.choice(prompt_parameters[chosen_params[-1]])}"
        prompt, ordered_details = prompt_function(prompt, all_details, chosen_detail, order)
        prompts.append(prompt)
        attributes.append(ordered_details)
    
    return prompts, attributes

features_to_template = ['name', 'birthdate', 'birth_city', 'university', 'employer', 'parent', 'child', 'best_friend']

def get_name_sentence_prompt(prompt_prefix: str, all_details: str, chosen_detail: str, order: int) -> str:
    ordered_details = ['name', chosen_detail]
    prompt = prompt_prefix
    prompt += " for introducing the name in a person's biography, written in third-person."
    prompt += " The template should have exactly one wildcard slot consisting of the word 'name' enclosed in curly braces."
    prompt += f" and exactly one wildcard slot for {all_details[chosen_detail]}, consisting of '{chosen_detail}' enclosed in curly braces."
    prompt += f" {ordered_details[order]} should come first and {ordered_details[1-order]} should come second in the template."
    prompt += f" always use the pronoun 'they' or 'them' to refer to the subject, without brackets, never call them 'subject'."
    prompt += " The template should be as short as possible, respecting the above constraints."
    return prompt, ordered_details

def other_sentence_prompt_generator(short_detail: str, long_detail: str):
    def get_other_sentence_prompt(prompt_prefix: str, *args) -> str:
        prompt = prompt_prefix
        prompt += f" for introducing the {long_detail} in a person's biography, written in third-person."
        prompt += f" The template should have exactly one wildcard slot for {long_detail}, consisting of '{short_detail}' enclosed in curly braces."
        prompt += " The template should be as short as possible, respecting the above constraints."
        return prompt, (short_detail,)
    return get_other_sentence_prompt

def generate_diverse_prompts(prompt_function, num_prompts=50):
    prompt_parameters = {
        'writing_style': ['concise', 'conversational'],
        'tone': ['casual', 'humorous', 'serious'],
        'sophistication': ['highbrow', 'lowbrow'],
        'formality': ['formal', 'informal'],
        'genre': ['academic', 'journalistic', 'technical'],
        'figurative_language': ['literal', 'idiomatic'],
        'emotion': ['neutral', 'excited'],
        'vocabulary_level': ['basic', 'intermediate', 'advanced'],
    }
    all_details = {'birth date': 'subject\'s date of birth',
               'birth city': 'subject\'s city of birth',
               'university': 'university the subject attended',
               'employer': 'subject\'s current employer'}
    
    prompts = []
    attributes = []
    for _ in range(num_prompts):
        chosen_params = random.sample(list(prompt_parameters.keys()), 3)
        chosen_detail = random.choice(list(all_details.keys()))
        order = random.choice([0, 1])
        ordered_details = ['name', chosen_detail]
        prompt = "Generate a brief sentence template that is "
        prompt += ", ".join([f"{random.choice(prompt_parameters[k])}" for k in chosen_params[:-1]])
        prompt += f" and {random.choice(prompt_parameters[chosen_params[-1]])}"
        prompt, ordered_details = prompt_function(prompt, all_details, chosen_detail, order)
        prompts.append(prompt)
        attributes.append(ordered_details)
    
    return prompts, attributes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate biography sentence templates")
    parser.add_argument("--detail", default="all", help="Specific detail to generate templates for (default: all)")
    args = parser.parse_args()

    all_details = {
        'name': 'subject\'s name',
        'birth_date': 'subject\'s date of birth',
        'birth_city': 'subject\'s city of birth',
        'university': 'university the subject attended',
        'employer': 'subject\'s current employer',
        'best_friend': 'subject\'s best friend',
        'parent': 'subject\'s parent',
        'child': 'subject\'s child',
        'worst_enemy': 'subject\'s worst enemy',
    }

    base_path = get_project_root()
    # Create the templates directory relative to the script location
    os.makedirs(base_path / "generated_data" / "templates", exist_ok=True)

    def generate_templates(detail, description):
        if detail == 'name':
            prompt_fn = get_name_sentence_prompt
        else:
            prompt_fn = other_sentence_prompt_generator(detail, description)

        diverse_prompts, attributes_list = generate_diverse_prompts(prompt_fn, num_prompts=250)
        templates = [
            template for prompt, attributes in zip(diverse_prompts, attributes_list)
            for template in query_llm(prompt)
            if verify_template(template, attributes)
        ]
        return filter_templates(templates)

    if args.detail == "all":
        for detail, description in all_details.items():
            best_templates = generate_templates(detail, description)
            
            filename = base_path / "generated_data" / "templates" / f"{detail}_templates.txt"
            with open(filename, "w") as f:
                for template in best_templates:
                    f.write(f"{template}\n")
            
            print(f"Generated and saved templates for {detail}")
    elif args.detail in all_details:
        best_templates = generate_templates(args.detail, all_details[args.detail])
        
        filename = base_path / "generated_data" / "templates" / f"{args.detail}_templates.txt"
        with open(filename, "w") as f:
            for template in best_templates:
                f.write(f"{template}\n")
        
        print(f"Generated and saved templates for {args.detail}")
    else:
        print(f"Invalid detail: {args.detail}. Choose from {', '.join(all_details.keys())} or 'all'.")
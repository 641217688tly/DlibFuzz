import os
import sys
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY", "")
)


def send_to_openai(prompt_template: str, content_to_be_annotated: str):
    message = [
        {
            'role': 'system',
            'content': prompt_template
        },
        {
            'role': 'user',
            'content': 'Now, please label the following issue:' + content_to_be_annotated
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=message,
        temperature=0.0
    )
                
    chat_response = response.choices[0].message.content

    return chat_response

def annotate_issues(prompt_template:str, issues: dict):
    annotated = {}
    for key, value in issues.items():
        print(f"Annotating issue {key}...")
        annotated[key] = send_to_openai(prompt_template, value)
    return annotated


def save_to_file(directory: str, prefix: str, title: str, item: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_dir = os.path.join(directory, prefix)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = f"{prefix}_{title}.json"
    file_path = os.path.join(file_dir, file_name)
    with open(file_path, 'w') as file:
        file.write(item)


if __name__ == "__main__":
    index_fetch_results = str(sys.argv[1])
    print(index_fetch_results)
    if not os.path.exists(f'results_{index_fetch_results}'):
        print(f"results_{index_fetch_results} does not exist.")
        exit(1)

    save_directory = f'results_{index_fetch_results}_annotated'

    # read the prompt template
    with open('prompt_template.txt', 'r') as f:
        prompt_template = f.read()
    print("Prompt template read.")
    
    # read the issues from the files and annotate them, then save them to designated files
    # PyTorch issues
    print("Reading PyTorch issues...")
    issues_torch = {}
    for file in os.listdir(f'results_{index_fetch_results}/pytorch_issue'):
        with open(f'results_{index_fetch_results}/pytorch_issue/{file}', 'r') as f:
            content = f.read()
            issues_torch[os.path.splitext(file)[0]] = content

    print("Annotating PyTorch issues...")
    annotated_torch = annotate_issues(prompt_template.format("PyTorch", "PyTorch", "PyTorch"), issues_torch)

    print("Saving annotated PyTorch issues...")
    index_pytorch_issues = 0
    for issue in annotated_torch.values():
        save_to_file(save_directory, 'pytorch_issue', str(index_pytorch_issues), issue)
        index_pytorch_issues += 1

    # TensorFlow issues
    print("Reading TensorFlow issues...")
    issues_tf = {}
    for file in os.listdir(f'results_{index_fetch_results}/tensorflow_issue'):
        with open(f'results_{index_fetch_results}/tensorflow_issue/{file}', 'r') as f:
            content = f.read()
            issues_tf[os.path.splitext(file)[0]] = content
    
    print("Annotating TensorFlow issues...")
    annotated_tf = annotate_issues(prompt_template.format("TensorFlow", "TensorFlow", "TensorFlow"), issues_tf)

    print("Saving annotated TensorFlow issues...")
    index_tf_issues = 0
    for issue in annotated_tf.values():
        save_to_file(save_directory, 'tensorflow_issue', str(index_tf_issues), issue)
        index_tf_issues += 1

    # JAX issues
    print("Reading JAX issues...")
    issues_jax = {}
    for file in os.listdir(f'results_{index_fetch_results}/jax_issue'):
        with open(f'results_{index_fetch_results}/jax_issue/{file}', 'r') as f:
            content = f.read()
            issues_jax[os.path.splitext(file)[0]] = content
    
    print("Annotating JAX issues...")
    annotated_jax = annotate_issues(prompt_template.format("Jax", "Jax", "Jax"), issues_jax)

    print("Saving annotated JAX issues...")
    index_jax_issues = 0
    for issue in annotated_jax.values():
        save_to_file(save_directory, 'jax_issue', str(index_jax_issues), issue)
        index_jax_issues += 1
    
    print("Done!")

import os
import sys
import asyncio
from aiohttp import ClientSession
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt


load_dotenv()
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY", "")
)

MAX_CONCURRENT_REQUESTS = 5
DELAY_BETWEEN_REQUESTS = 1.5


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
        model="gpt-4o-mini",
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


def annotate_and_save_issues(prompt_template: str, issues: dict, directory: str, prefix: str):
    annotated = {}
    for key, value in issues.items():
        print(f"Annotating issue {key}...")
        annotated[key] = send_to_openai(prompt_template, value)
        save_to_file(directory, prefix, key, annotated[key])
    return annotated


async def send_to_openai_async(prompt_template: str, 
                               content_to_be_annotated: str, 
                               semaphore: asyncio.Semaphore
                               ) -> str:
    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
    async def _send_request():
        async with semaphore:
            await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
            async with ClientSession() as session:
                response = await send_to_openai(session, prompt_template, content_to_be_annotated)
                return response
    return await _send_request()
    

async def process_single_issue(key: str, 
                               value: str, 
                               prompt_template: str, 
                               directory: str, 
                               prefix: str, 
                               semaphore: asyncio.Semaphore
                               ) -> str:
    try:
        annotated = await send_to_openai_async(prompt_template, value, semaphore)
        save_to_file(directory, prefix, key, annotated)
        return key, annotated
    except Exception as e:
        print(f"Failed to annotate issue {key}: {str(e)}")
        return key, None
    

async def annotate_and_save_issues_async(prompt_template: str, 
                                         issues: dict, 
                                         directory: str, 
                                         prifix: str
                                         ) -> dict:
    annotated = {}
    tasks = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    for key, value in issues.items():
        print(f"Annotating issue {key}...")
        task = asyncio.create_task(
            process_single_issue(key, value, prompt_template, directory, prifix, semaphore)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)

    for key, result in results:
        annotated[key] = result if result else "Failed to annotate issue."
    
    return annotated



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
    # annotated_torch = annotate_and_save_issues(prompt_template.format("PyTorch", "PyTorch", "PyTorch"), 
    #                                            issues_torch, 
    #                                            save_directory, 
    #                                            'pytorch_issue'
    #                                            )
    asyncio.run(annotate_and_save_issues_async(prompt_template.format("PyTorch", "PyTorch", "PyTorch"), 
                                               issues_torch, 
                                               save_directory, 
                                               'pytorch_issue'
                                               ))

    # JAX issues
    print("Reading JAX issues...")
    issues_jax = {}
    for file in os.listdir(f'results_{index_fetch_results}/jax_issue'):
        with open(f'results_{index_fetch_results}/jax_issue/{file}', 'r') as f:
            content = f.read()
            issues_jax[os.path.splitext(file)[0]] = content
    
    print("Annotating JAX issues...")
    # annotated_jax = annotate_and_save_issues(prompt_template.format("Jax", "Jax", "Jax"), 
    #                                          issues_jax, 
    #                                          save_directory, 
    #                                          'jax_issue'
    #                                          )
    asyncio.run(annotate_and_save_issues_async(prompt_template.format("Jax", "Jax", "Jax"), 
                                               issues_jax, 
                                               save_directory, 
                                               'jax_issue'
                                               ))

    # print("Saving annotated JAX issues...")
    # index_jax_issues = 0
    # for issue in annotated_jax.values():
    #     save_to_file(save_directory, 'jax_issue', str(index_jax_issues), issue)
    #     index_jax_issues += 1

    # MindSpore issues
    print("Reading MindSpore issues...")
    issues_ms = {}
    for file in os.listdir(f'results_{index_fetch_results}/ms_issue'):
        with open(f'results_{index_fetch_results}/ms_issue/{file}', 'r') as f:
            content = f.read()
            issues_ms[os.path.splitext(file)[0]] = content
    
    print("Annotating MindSpore issues...")
    # annotated_ms = annotate_and_save_issues(prompt_template.format("MindSpore", "MindSpore", "MindSpore"), 
    #                                         issues_ms, 
    #                                         save_directory, 
    #                                         'ms_issue'
    #                                         )
    asyncio.run(annotate_and_save_issues_async(prompt_template.format("MindSpore", "MindSpore", "MindSpore"), 
                                               issues_ms, 
                                               save_directory, 
                                               'ms_issue'
                                               ))

    # print("Saving annotated MindSpore issues...")
    # index_ms_issues = 0
    # for issue in annotated_ms.values():
    #     save_to_file(save_directory, 'ms_issue', str(index_ms_issues), issue)
    #     index_ms_issues += 1
    
    print("Done!")

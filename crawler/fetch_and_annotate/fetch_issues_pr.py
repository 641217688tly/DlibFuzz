import os
import time
import asyncio
import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import datetime
from dotenv import load_dotenv


async def create_session_with_retries():
    timeout = aiohttp.ClientTimeout(total=120)
    connector = aiohttp.TCPConnector(limit=10)  # limit concurrent connections
    session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    return session


import time

async def handle_rate_limiting(response):
    # Check for rate limit status and handle accordingly
    if response.status in {429, 403} or 'X-RateLimit-Remaining' in response.headers:
        # Retrieve the remaining requests count
        rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
        
        # If remaining requests are 0, we need to wait until reset time
        if rate_limit_remaining == 0:
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time()))
            sleep_time = reset_time - int(time.time()) + 1  # Calculate remaining time until reset
            print(f"Rate limit reached. Sleeping for {sleep_time} seconds.")
            await asyncio.sleep(sleep_time)
        else:
            print(f"Remaining requests: {rate_limit_remaining}")
    else:
        print("No rate limiting headers found, proceeding without delay.")


# # Check and handle rate limiting
# async def handle_rate_limiting(response):
#     if response.status == 403 and 'X-RateLimit-Remaining' in response.headers:
#         rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
#         if rate_limit_remaining == 0:
#             reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
#             sleep_time = reset_time - int(time.time()) + 1
#             print(f"Rate limit reached. Sleeping for {sleep_time} seconds.")
#             await asyncio.sleep(sleep_time)


async def fetch_issues(repo_owner: str, repo_name: str, label: str='bug', num_results: int=100, session: ClientSession=None) -> list[dict]:
    issues = []
    page = 1
    headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}'}
    tasks = []
    
    while len(issues) < num_results:
        url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/issues'
        params = {
            'state': 'all',
            'labels': label,
            'page': page,
            'per_page': 100
        }
        async with session.get(url, headers=headers, params=params) as response:
            await handle_rate_limiting(response)
            
            if response.status != 200:
                raise Exception(f"Failed to fetch issues: {response.status}")
            
            page_issues = await response.json()
            if not page_issues:
                break

            for issue in page_issues:
                title = issue.get('title')
                issue_url = issue.get('html_url')
                state = issue.get('state')
                # Create a task for fetching content asynchronously
                task = asyncio.create_task(fetch_issue_content(issue_url, session))
                tasks.append((title, issue_url, state, task))
                
                if len(tasks) >= num_results:
                    break
            page += 1
    
    # Await all content fetch tasks concurrently
    for title, issue_url, state, task in tasks:
        content = await task
        issues.append({'title': title, 'url': issue_url, 'state': state, 'content': content})

    return issues


async def fetch_pull_requests(repo_owner: str, repo_name: str, state: str='open', num_results: int=100, session: ClientSession=None) -> list[dict]:
    pull_requests = []
    page = 1
    headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}'}
    tasks = []
    
    while len(pull_requests) < num_results:
        url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/pulls'
        params = {
            'state': state,
            'page': page,
            'per_page': 100
        }
        async with session.get(url, headers=headers, params=params) as response:
            await handle_rate_limiting(response)

            if response.status != 200:
                raise Exception(f"Failed to fetch pull requests: {response.status}")
            
            page_pull_requests = await response.json()
            if not page_pull_requests:
                break

            for pr in page_pull_requests:
                title = pr.get('title')
                pr_url = pr.get('html_url')
                # Create a task for fetching content asynchronously
                task = asyncio.create_task(fetch_pr_content(pr_url, session))
                tasks.append((title, pr_url, task))
                
                if len(tasks) >= num_results:
                    break
            page += 1
    
    for title, pr_url, task in tasks:
        content = await task
        pull_requests.append({'title': title, 'url': pr_url, 'content': content})

    return pull_requests


async def fetch_issue_content(issue_url: str, session: ClientSession, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            async with session.get(issue_url) as response:
                await handle_rate_limiting(response)

                if response.status != 200:
                    return 'Failed to fetch issue content.'
                
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                content_div = soup.find('div', {'class': 'edit-comment-hide'})
                content = content_div.text.strip() if content_div else 'No content found...'
                return content
        except asyncio.TimeoutError:
            if attempt < retries - 1:
                backoff = (2 ** attempt)
                print(f"Timeout occurred. Retrying in {backoff} seconds...")
                await asyncio.sleep(backoff)
            else:
                return 'Timeout occurred.'


async def fetch_pr_content(pr_url: str, session: ClientSession, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            async with session.get(pr_url) as response:
                await handle_rate_limiting(response)

                if response.status != 200:
                    return 'Failed to fetch pull request content.'
                
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                content_div = soup.find('div', {'class': 'comment-body'})
                content = content_div.text.strip() if content_div else 'No content found...'
                return content
        except asyncio.TimeoutError:
            if attempt < retries - 1:
                backoff = (2 ** attempt)
                print(f"Timeout occurred. Retrying in {backoff} seconds...")
                await asyncio.sleep(backoff)
            else:
                return 'Timeout occurred.'



def save_to_file(directory: str, prefix: str, title: str, item: dict):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_dir = os.path.join(directory, prefix)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = f"{prefix}_{title}.txt"
    file_path = os.path.join(file_dir, file_name)
    with open(file_path, 'w') as file:
        file.write(f"Title: {item['title']}\n")
        file.write(f"URL: {item['url']}\n")
        file.write(f"Content: {item['content']}\n")


async def main():
    load_dotenv()
    current_time = datetime.datetime.now()
    current_time_in_str = current_time.strftime('%m%d%H%M%S')
    save_directory = f'results_{current_time_in_str}'

    session = await create_session_with_retries()

    async with session:
        print('Fetching issues and pull requests from the target repositories...')

        # fetch issues and PRs concurrently for each repository
        issues_torch_task = fetch_issues('pytorch', 'pytorch', num_results=1000, session=session)
        pr_torch_task = fetch_pull_requests('pytorch', 'pytorch', num_results=1000, session=session)
        issues_tf_task = fetch_issues('tensorflow', 'tensorflow', label='type:bug', num_results=1000, session=session)
        pr_tf_task = fetch_pull_requests('tensorflow', 'tensorflow', num_results=1000, session=session)
        issues_jax_task = fetch_issues('google', 'jax', num_results=1000, session=session)
        pr_jax_task = fetch_pull_requests('google', 'jax', num_results=1000, session=session)

        issues_torch, pr_torch, issues_tf, pr_tf, issues_jax, pr_jax = await asyncio.gather(
            issues_torch_task, pr_torch_task,
            issues_tf_task, pr_tf_task,
            issues_jax_task, pr_jax_task
        )
        
        print('Saving results for PyTorch...')
        for idx, issue in enumerate(issues_torch):
            save_to_file(save_directory, 'pytorch_issue', str(idx), issue)
        for idx, pr in enumerate(pr_torch):
            save_to_file(save_directory, 'pytorch_pr', str(idx), pr)

        print('Saving results for TensorFlow...')
        for idx, issue in enumerate(issues_tf):
            save_to_file(save_directory, 'tensorflow_issue', str(idx), issue)
        for idx, pr in enumerate(pr_tf):
            save_to_file(save_directory, 'tensorflow_pr', str(idx), pr)

        print('Saving results for JAX...')
        for idx, issue in enumerate(issues_jax):
            save_to_file(save_directory, 'jax_issue', str(idx), issue)
        for idx, pr in enumerate(pr_jax):
            save_to_file(save_directory, 'jax_pr', str(idx), pr)



if __name__ == "__main__":
    asyncio.run(main())

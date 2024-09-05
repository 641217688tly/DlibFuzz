import datetime
import os
import requests
from bs4 import BeautifulSoup


def fetch_issues(repo_owner: str, repo_name: str, label: str='bug', num_results: int=100) -> list[dict]:
    issues = []
    page = 1
    headers = {'Authorization': os.getenv('GITHUB_TOKEN', '')}

    while len(issues) < num_results:
        url =f'https://api.github.com/repos/{repo_owner}/{repo_name}/issues'
        params = {
            'state': 'open',
            'labels': label,
            'page': page,
            'per_page': 100
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch issues: {response.status_code}")

        page_issues = response.json()
        if not page_issues:
            break

        for issue in page_issues:
            title = issue.get('title')
            issue_url = issue.get('html_url')
            state = issue.get('state')
            content = fetch_issue_content(issue_url)
            issues.append({'title': title, 'url': issue_url, 'state':state, 'content': content})
            if len(issues) >= num_results:
                break
        page += 1

    
    return issues


def fetch_pull_requests(repo_owner: str, repo_name: str, state: str='open', num_results: int=100) -> list[dict]:
    pull_requests = []
    page = 1
    while len(pull_requests) < num_results:
        url = f'https://github.com/{repo_owner}/{repo_name}/pulls?q=is%3Apr+is%3A{state}&page={page}&per_page=30'
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch pull requests: {response.status_code}")

        soup = BeautifulSoup(response.text, 'html.parser')
        pr_rows = soup.find_all('div', {'class': 'js-issue-row'})
        if not pr_rows:
            break  # Exit if no more pull requests are found

        for pr in pr_rows:
            title = pr.find('a', {'data-hovercard-type': 'pull_request'}).text.strip()
            pr_url = 'https://github.com' + pr.find('a', {'data-hovercard-type': 'pull_request'})['href']
            content = fetch_pr_content(pr_url)
            pull_requests.append({'title': title, 'url': pr_url, 'content': content})
            if len(pull_requests) >= num_results:
                break
        page += 1
    
    return pull_requests


def fetch_issue_content(issue_url: str) -> str:
    response = requests.get(issue_url)
    if response.status_code != 200:
        return 'Failed to fetch issue content.'
    
    soup = BeautifulSoup(response.text, 'html.parser')
    content_div = soup.find('div', {'class': 'edit-comment-hide'})
    content = content_div.text.strip() if content_div else 'No content found...'
    return content


def fetch_pr_content(pr_url: str) -> str:
    response = requests.get(pr_url)
    if response.status_code != 200:
        return 'Failed to fetch pull request content.'
    
    soup = BeautifulSoup(response.text, 'html.parser')
    content_div = soup.find('div', {'class': 'comment-body'})
    content = content_div.text.strip() if content_div else 'No content found...'
    return content


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


if __name__ == "__main__":
    current_time = datetime.datetime.now()
    current_time_in_str = current_time.strftime('%m%d%H%M%S')

    save_directory = f'results_{current_time_in_str}'

    # fetch issues and pull requests from PyTorch
    print('Fetching issues from PyTorch...')
    issues_torch = fetch_issues('pytorch', 'pytorch', num_results=100)

    print('Saving the results to "pytorch_issue"...')
    index_pytorch_issues = 0
    for issue in issues_torch:
        save_to_file(save_directory, 'pytorch_issue', str(index_pytorch_issues), issue)
        index_pytorch_issues += 1
    
    print('Fetching pull requests from PyTorch...')
    pr_torch = fetch_pull_requests('pytorch', 'pytorch', num_results=100)

    print('Saving the results to "pytorch_pr"...')
    index_pytorch_pr = 0
    for pr in pr_torch:
        save_to_file(save_directory, 'pytorch_pr', str(index_pytorch_pr), pr)
        index_pytorch_pr += 1
    
    # fetch issues and pull requests from TensorFlow
    print('Fetching issues from TensorFlow...')
    issues_tf = fetch_issues('tensorflow', 'tensorflow', label='type:bug', num_results=100)

    print('Saving the results to "tensorflow_issue"...')
    index_tf_issues = 0
    for issue in issues_tf:
        save_to_file(save_directory, 'tensorflow_issue', str(index_tf_issues), issue)
        index_tf_issues += 1
    
    print('Fetching pull requests from TensorFlow...')
    pr_tf = fetch_pull_requests('tensorflow', 'tensorflow', num_results=100)

    print('Saving the results to "tensorflow_pr"...')
    index_tf_pr = 0
    for pr in pr_tf:
        save_to_file(save_directory, 'tensorflow_pr', str(index_tf_pr), pr)
        index_tf_pr += 1
    
    # fetch issues and pull requests from JAX
    print('Fetching issues from JAX...')
    issues_jax = fetch_issues('google', 'jax', num_results=100)

    print('Saving the results to "jax_issue"...')
    index_jax_issues = 0
    for issue in issues_jax:
        save_to_file(save_directory, 'jax_issue', str(index_jax_issues), issue)
        index_jax_issues += 1
    
    print('Fetching pull requests from JAX...')
    pr_jax = fetch_pull_requests('google', 'jax', num_results=100)

    print('Saving the results to "jax_pr"...')
    index_jax_pr = 0
    for pr in pr_jax:
        save_to_file(save_directory, 'jax_pr', str(index_jax_pr), pr)
        index_jax_pr += 1

    print('Done!')

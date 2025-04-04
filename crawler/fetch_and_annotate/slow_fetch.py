import datetime
import os
import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib3 import Retry

load_dotenv()

def create_session_with_retries():
    session = requests.Session()
    retry = Retry(
        total=5,  # Retry up to 5 times
        backoff_factor=1,  # Wait 1 second between retries, then 2, 4, etc.
        status_forcelist=[429, 500, 502, 503, 504]  # Retry on these HTTP statuses
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

session = create_session_with_retries()


def fetch_issues(saving_directory: str, 
                 dir_prefix: str, 
                 repo_owner: str, 
                 repo_name: str, 
                 label: str='bug', 
                 num_results: int=100
                 ) -> list[dict]:
    issues = []
    page = 1
    headers = {'Authorization': os.getenv('GITHUB_TOKEN', '')}
    print(f"token: {os.getenv('GITHUB_TOKEN', '')}")


        

    while len(issues) < num_results:
        url =f'https://api.github.com/repos/{repo_owner}/{repo_name}/issues'
        if label == 'all':
            params = {
                'state': 'all',
                'page': page,
                'per_page': 100
            }
        else:
            params = {
                'state': 'all',
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
            save_to_file(saving_directory, dir_prefix, len(issues), {'title': title, 
                                                                     'url': issue_url, 
                                                                     'state': state, 
                                                                     'content': content
                                                                     })
            issues.append({'title': title, 
                           'url': issue_url, 
                           'state':state, 
                           'content': content
                           })
            if len(issues) >= num_results:
                break
        page += 1

    
    return issues


def fetch_pull_requests(saving_directory: str, 
                        dir_prefix: str, 
                        repo_owner: str, 
                        repo_name: str, 
                        state: str='open', 
                        num_results: int=100
                        ) -> list[dict]:
    pull_requests = []
    page = 1
    headers = {'Authorization': os.getenv('GITHUB_TOKEN', '')}

    while len(pull_requests) < num_results:
        url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/pulls'
        # url = f'https://github.com/{repo_owner}/{repo_name}/pulls?q=is%3Apr+is%3A{state}&page={page}&per_page=30'
        params = {
            'state': state,
            'page': page,
            'per_page': 100
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch pull requests: {response.status_code}")

        page_pull_requests = response.json()
        if not page_pull_requests:
            break

        for pr in page_pull_requests:
            title = pr.get('title')
            pr_url = pr.get('html_url')
            content = fetch_pr_content(pr_url)
            save_to_file(saving_directory, dir_prefix, len(pull_requests), {'title': title, 
                                                                            'url': pr_url, 
                                                                            'content': content
                                                                            })
            pull_requests.append({'title': title, 
                                  'url': pr_url, 
                                  'content': content
                                  })
            if len(pull_requests) >= num_results:
                break
        page += 1

        return pull_requests
    

def fetch_issues_gitee(saving_directory: str, 
                       dir_prefix: str, 
                       repo_owner: str, 
                       repo_name: str, 
                       label: str='bug', 
                       num_results: int=100
                       ) -> list[dict]:
    issues = []
    page = 1
    headers = {'Authorization': f"Bearer {os.getenv('GITEE_TOKEN', '')}"}
    print(f"token: {os.getenv('GITEE_TOKEN', '')}")

    while len(issues) < num_results:
        url = f'https://gitee.com/api/v5/repos/{repo_owner}/{repo_name}/issues'
        if label == 'all':
            params = {
                'state': 'all',
                'page': page,
                'per_page': 100,
                'sort': 'created',
                'direction': 'desc'
            }
        else:
            params = {
                'state': 'all',
                'labels': label,
                'page': page,
                'per_page': 100,
                'sort': 'created',
                'direction': 'desc'
            }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch issues: {response.status_code}, Response: {response.text}")

        page_issues = response.json()
        if not page_issues:
            break

        for issue in page_issues:
            title = issue.get('title')
            issue_url = issue.get('html_url')
            state = issue.get('state')
            content = issue.get('body', '')
            save_to_file(saving_directory, dir_prefix, len(issues), {'title': title, 
                                                                     'url': issue_url, 
                                                                     'state': state, 
                                                                     'content': content
                                                                     })
            issues.append({'title': title, 
                           'url': issue_url, 
                           'state': state, 
                           'content': content
                           })
            if len(issues) >= num_results:
                break
        page += 1

    return issues


def fetch_issue_content(issue_url: str) -> str:
    response = session.get(issue_url)
    if response.status_code != 200:
        return 'Failed to fetch issue content.'
    
    soup = BeautifulSoup(response.text, 'html.parser')
    content_div = soup.find('div', {'class': 'edit-comment-hide'})
    content = content_div.text.strip() if content_div else 'No content found...'
    return content


def fetch_pr_content(pr_url: str) -> str:
    response = session.get(pr_url)
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
    print('Fetching issues labelled "bug" from PyTorch...')
    issues_torch = fetch_issues(saving_directory=save_directory, 
                                dir_prefix='pytorch_issues', 
                                repo_owner='pytorch', 
                                repo_name='pytorch', 
                                num_results=5000, 
                                label='bug'
                                )

    print('Fetching issues labelled "high priority" from PyTorch...')
    issues_torch_high_priority = fetch_issues(saving_directory=save_directory, 
                                              dir_prefix='pytorch_issues_hp', 
                                              repo_owner='pytorch', 
                                              repo_name='pytorch', 
                                              num_results=5000, 
                                              label='high priority'
                                              )


    # print('Fetching pull requests from PyTorch...')
    # pr_torch = fetch_pull_requests('pytorch', 'pytorch', num_results=1000)

    # print('Saving the results to "pytorch_pr"...')
    # index_pytorch_pr = 0
    # for pr in pr_torch:
    #     save_to_file(save_directory, 'pytorch_pr', str(index_pytorch_pr), pr)
    #     index_pytorch_pr += 1


    # fetch issues and pull requests from JAX
    print('Fetching issues from JAX...')
    issues_jax = fetch_issues(saving_directory=save_directory, 
                              dir_prefix='jax_issues', 
                              repo_owner='google', 
                              repo_name='jax', 
                              num_results=5000
                              )


    # print('Fetching pull requests from JAX...')
    # pr_jax = fetch_pull_requests('google', 'jax', num_results=1000)

    # print('Saving the results to "jax_pr"...')
    # index_jax_pr = 0
    # for pr in pr_jax:
    #     save_to_file(save_directory, 'jax_pr', str(index_jax_pr), pr)
    #     index_jax_pr += 1

    # fetch issues and pull requests from MindSpore
    print('Fetching issues from MindSpore...')
    issues_ms = fetch_issues(saving_directory=save_directory, 
                             dir_prefix='mindspore_issues', 
                             repo_owner='mindspore-ai', 
                             repo_name='mindspore', 
                             num_results=5000, 
                             label='all'
                             )


    # fetch issues from MindSpore's Gitee repository
    print('Fetching issues from MindSpore...')
    issues_ms_gitee = fetch_issues_gitee(saving_directory=save_directory, 
                                         dir_prefix='mindspore_issues_gitee', 
                                         repo_owner='mindspore', 
                                         repo_name='mindspore', 
                                         num_results=5000, 
                                         label='all'
                                         )


    # print('Fetching pull requests from MindSpore...')
    # pr_ms = fetch_pull_requests('mindspore-ai', 'mindspore', num_results=1000)

    # print('Saving the results to "ms_pr"...')
    # index_ms_pr = 0
    # for pr in pr_ms:
    #     save_to_file(save_directory, 'ms_pr', str(index_ms_pr), pr)
    #     index_ms_pr += 1

 
    # fetch issues and pull requests from Jittor
    print('Fetching issues from Jittor...')
    issues_jt = fetch_issues(saving_directory=save_directory, 
                             dir_prefix='jittor_issues', 
                             repo_owner='Jittor', 
                             repo_name='jittor', 
                             num_results=5000, 
                             label='all'
                             )


    # print('Fetching pull requests from Jittor...')
    # pr_jt = fetch_pull_requests('Jittor', 'jittor', num_results=1000)

    # print('Saving the results to "jt_pr"...')
    # index_jt_pr = 0
    # for pr in pr_jt:
    #     save_to_file(save_directory, 'ms_jt', str(index_jt_pr), pr)
    #     index_ms_jt += 1


    print('Done!')


import pip
 
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])


import_or_install('PyGithub')
from github import Github

import requests

import json
import os
 
# Change to your ACCESS_TOKEN
ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN'

g = Github(ACCESS_TOKEN)
repo = g.get_repo('Illania/Datasets')
file_content = repo.get_contents('twitter.csv')
download_url = file_content.download_url
response = requests.get(download_url)

with open('/home/illania/project/datasets/data.csv', 'wb') as file:
    file.write(response.content)

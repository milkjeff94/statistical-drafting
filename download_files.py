import requests

url = "https://17lands-public.s3.amazonaws.com/analysis_data/cards/abilities.csvzzz"
local_filename = "draft_data_public.FDN.PremierDraft.csv.gz"

# Send GET request to fetch the file
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    with open(local_filename, 'wb') as file:
        file.write(response.content)
    print(f"File successfully downloaded: {local_filename}")
else:
    print(f"Failed to download file. Status code: {response.status_code}")
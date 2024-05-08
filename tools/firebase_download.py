import requests
import json
import datetime
import os

# Set the API key in env
os.environ["API_KEY"] = "aFwkNaGuqvEIHtP-M6LQDTNevSF=XfPMJm-GxVT/osOMZwn8HIZEZGH?64S7IwPB"

def download_latest_videos(size = 2, start_timestamp = None, end_timestamp = None):
    # Calculate the timestamp for the start and end of the last month
    now = datetime.datetime.now()
    month_ago = now - datetime.timedelta(days=30)
    day_ago = now - datetime.timedelta(days=1)
    
    if start_timestamp is None:
        start_timestamp = int(month_ago.timestamp() * 1000)
    if end_timestamp is None:
        end_timestamp = int(now.timestamp() * 1000)
    
    # Define the API endpoint and the request body
    
    url = "https://us-central1-sportsbox-3dgolf.cloudfunctions.net/api/elastic/search/video"

    body = {
        "size": size,
        "date": {"gt": start_timestamp, "lte": end_timestamp},
        "dominantHand": "Right",
        "exclude_ssl2DTag": ["2DSSL_Invalid", "2DSSL_Labeled", "2DSSL_Trained", "2DSSL_Good", "2DSSL_Improve"]
        # "ssl2DTag": "2DSSL_Requested"
    }

    headers = {
    "Authorization": os.environ["API_KEY"],
    }

    # Make the POST request
    response = requests.post(url, json=body, headers=headers)
    
    # Check the response status code
    if response.status_code != 200:
        print(f"Error: {response.content}")
        raise ValueError(f"Error: {response.content}")
    
    # Return the response content in JSON format
    return response.json()


import json

def clean_firebase_response(response):
    cleaned = {}
    cleaned["pit_id"] = response["pit_id"]
    cleaned["took"] = response["took"]
    cleaned["timed_out"] = response["timed_out"]
    cleaned["total_hits"] = response["hits"]["total"]["value"]

    hits = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        source["metaData"]["measurement"].update(source["userData"])
        del source["actionIds"]
        del source["userData"]
        hits.append(source)

    cleaned["hits"] = hits

    return json.dumps(cleaned)

import csv

def extract_data_to_csv(response, output_path):
    data = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        row = []
        row.append(source["videoPath"])
        row.append(source["videoOrigName"])
        row.append(source["videoType"])
        row.append(source["videoSource"])
        row.append(source["doc_relations"]["parent"])
        row.append(source["videoCreated"])
        if "measurement" in source["metaData"]:
            row.append(source["metaData"]["measurement"]["hipDistance"])
        else:
            row.append("")
        row.append(source["metaData"]["fps"])
        videoIds = []

        for session in hit["inner_hits"]["sessions"]["hits"]["hits"]:
            videoIds.extend(session['_source']["videoIds"])
        row.append(videoIds)
        
        if "analysis" in hit["inner_hits"]:
            try:
                analysis = hit["inner_hits"]["analysis"]["hits"]["hits"][0]["_source"]
                row.append(analysis["swingConfidenceScore"])
                row.append(analysis["swingScore"])
            except:
                row.append("")
                row.append("")
        else:
            row.append("")
            row.append("")
        data.append(row)
    print(f'Number of videos: {len(data)}')
    with open(output_path, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["videoPath","videoOrigName","videoType","videoSource","sessionName","videoCreated","hipDistance", "fps","videoIds", "swingConfidenceScore","swingScore"])
        csvwriter.writerows(data)
    

if __name__ == "__main__":
    # Download the latest videos
    videos = download_latest_videos()

    # Print the videos
    print(json.dumps(videos, indent=4))

    cleaned_response = clean_firebase_response(videos)

    print(cleaned_response)
    extract_data_to_csv(videos)

    # Save videos to output.json in readable format
    with open("output.json", "w") as f:
        json.dump(videos, f, indent=4)
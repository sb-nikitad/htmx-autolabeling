import requests

def tag_video(video_id, tag):
    
    # Define the API endpoint and the request body

    url = "https://us-central1-sportsbox-3dgolf.cloudfunctions.net/api/videosTag"
    body = {
    "videoIds": [video_id],
    "tag": tag
    }   

    # Make the PATCH request
    response = requests.patch(url, json=body)
    
    # Check the response status code
    if response.status_code != 200:
        print(f"Error: {response.content}")
        raise ValueError(f"Error: {response.content}")

    # Return the response content in JSON format
    print(response.json())
    return response.json()
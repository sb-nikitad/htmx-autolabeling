import requests
import os
import csv

def download_video_files_baseball(videos_csv_path, output_path, imported=False):
    with open(videos_csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if imported:
                video_path = row[-1]
                video_name = row[0]
                video_type = 'mov'
            else:
                video_path = row[0]
                video_name = row[1]
                video_type = row[2]
            # download video to local folder
            r = requests.get(video_path, allow_redirects=True)

            # Check response status
            if r.status_code != 200:
                print(f"Error downloading video {video_name}")
                continue
            try:
                with open(f'{os.path.join(output_path, video_name)}.{video_type}', 'wb') as f:
                    f.write(r.content)
            except:
                print(f"Error downloading video {video_name}")
                continue

def tag_videos_baseball(videoIds, videoTags):
    ### TODO: change to baseball url/auth
    url = "https://us-central1-sportsbox-3dgolf.cloudfunctions.net/api/videosTag"
    headers = {
        "Authorization": os.environ["API_KEY"],
    }

    for id, tag in zip(videoIds, videoTags):
        #TODO: load list directly
        id = eval(id)

        body = {
            "videoIDs": id,
            "tag": tag
        }

        # Make the PATCH request
        response = requests.patch(url, json=body, headers=headers)
        print(f'Tagging {id} with {tag}')
        print(response)
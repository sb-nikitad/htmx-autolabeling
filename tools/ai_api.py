import requests
import csv
import json
import os
from multiprocessing import Pool
import random
from tqdm import tqdm
import pandas as pd

def get_json(video_url):

    hip_length = 121

    fps = 120

    url = 'http://68.64.52.146:40030/web_inference/'
    myobj = {'videoUrl': video_url,
                'hipDistance': hip_length,
                'fps': fps,
                'smooth2d': False,
                'smooth3d': False,
                'smooth_type': 'butterworth',
                'model_2d_name': 'lpn_pose2d_tflite_38_140k_pretrained',
                'model_3d_name': 'attn_videopose3d_34_1M_deg0_90',
                'ws2d': 15,
                'ws3d': 15,
                'time': str(random.randint(0, 1000))}
    
    x = requests.post(url, data=json.dumps(myobj), headers = {"Content-Type": "text/plain; charset=utf-8"}, verify=False)
    
    # If response is not 200, print error
    if x.status_code != 200:
        print(f'Error: {x.status_code} - {video_url}')
        return None
    response = x.json()

    return response['jsonai']

def get_ai_from_csv(videos_csv_path, ai_path, imported=False, progress=None):
    # p = Pool(2)

    videos_list = pd.read_csv(videos_csv_path)

    with open(videos_csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        
        i=1
        for row in tqdm(reader):
            print(f'Downloading jsonai for {i} video')

            if progress is not None:
                progress(i/len(videos_list), desc="Downloading AI files")
            
            i+=1
            if imported:
                video_path = row[-1]
                video_name = row[0]
                video_type = 'mov'
            else:
                video_path = row[0]
                video_name = row[1]
                video_type = row[2]
            if os.path.exists(os.path.join(ai_path, video_name+f'.{video_type}' + "_ai.json")):
                print(f'File {video_name+f".{video_type}" + "_ai.json"} already exists')
                continue
            #Process files from folder
            
            # Remove the file extension
            # video_name = os.path.splitext('.')[0]
            try:
                with open(os.path.join(ai_path, video_name+f'.{video_type}' + "_ai.json"), 'w') as f:
                    # Write json to file
                    
                    jsonai= get_json(video_path)

                    if jsonai is None:
                        print(f'Error getting jsonai for {video_name}')
                        continue
                    print(f'Writing jsonai to file{os.path.join(ai_path, video_name + "_ai.json")}')
                    json.dump(jsonai, f)
            except:
                print(f'Error getting jsonai for {video_name}')
                continue

            # p.apply_async(get_json, args=(video_path,))
    
    # p.close()
    # p.join()

if __name__ == "__main__":
    p = Pool(2)

    #Process files from folder
    folder = "/mnt/nas01/Uploads/videos/SSL_Videos/Videos_Pipeline_prep/1_Score_0-80_20221101-1216"
    for file in os.listdir(folder):
        absolute_path = os.path.join(folder, file)
        
        p.apply_async(get_json, args=(absolute_path,))
    
    p.close()
    p.join()

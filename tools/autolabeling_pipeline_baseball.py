import requests
import os
import json
import csv
import shutil
import pandas as pd
import datetime
import gradio as gr

from ai_api import get_ai_from_csv
from firebase_download_baseball import download_latest_videos 
from autolabeling_baseball import videos_processing
from tqdm import tqdm
from logs import log_run, update_log
from api_3dbaseball import download_video_files_baseball, tag_videos_baseball

def run_pipeline(conf_thresh = 0.8, confidence_club = 0.7, date_start='', date_end='', size = 1, model_name = '', imported=False, tag_mode=False, keyword='', api=False, progress=gr.Progress()):
    output_dir='/mnt/nas01/Uploads/videos/SSL_Videos/Auto_Labelling_Results/'

    progress(0, desc="Downloading the videos")
    
    dstamp = datetime.datetime.now()
    run_date = dstamp.strftime("%Y%m%d%H%M")
    
    log_run(run_date, model_name, size, keyword, 'In progress')

    print(f'Imported: {imported}')
    #print("api:", api)
    # Download the latest videos

    # Convert the date utc timestamp
    if not imported:
        try:
            if date_start != '':
                date_start = int(datetime.datetime.strptime(date_start, '%Y-%m-%d').timestamp()) * 1000
                print(f'date_start: {date_start}')
            if date_end != '':
                date_end = int(datetime.datetime.strptime(date_end, '%Y-%m-%d').timestamp()) * 1000
                print(f'date_end: {date_end}')
        except Exception as e:
            print(f'Wrong date format: {e}')
            e = Exception('Wrong date format')
            raise e
            return 'Wrong date format'

    # Creating output directory
    output_dir = os.path.join(output_dir, 'AutoLabel_Web')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    print('Running the pipeline')
    print('Downloading the latest videos')
    if not  imported:
        videos = download_latest_videos(size=size, start_timestamp=date_start, end_timestamp=date_end)

    aiDir = os.path.join(output_dir, "ai")
    if not os.path.exists(aiDir):
        os.makedirs(aiDir)
    
    dataDir = os.path.join(output_dir, "videos")
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)
    else:
        shutil.rmtree(dataDir)
        os.makedirs(dataDir)

    output = os.path.join(output_dir, "output")
    if not os.path.exists(output):
        os.makedirs(output)
    
    model = "./models/Sportsbox_Baseball_Pose2D_FO_36_108K_20240206.tflite"
        
    if model_name != '':
        model = os.path.join('models', f'{model_name}.tflite')
        print(f'Using model {model}')
    
    progress(0.02, desc="Preparing data for processing")

    videos_csv = os.path.join(output_dir, "videos.csv")
    if not imported:
        print('Extracting data to csv')
        extract_data_to_csv(videos, videos_csv)
    else:
        if api:
            videos = pd.read_csv(imported)
        else:
            videos = pd.read_csv(imported.name)
        videos.to_csv(videos_csv, index=False)

    print('Getting AI data from csv')
    progress(0.05, desc="Downloading AI files")
    get_ai_from_csv(videos_csv, aiDir, imported, progress)
    print(f'Number of AI files: {len(os.listdir(aiDir))}')
    progress(0.10, desc="Downloading video files")
    print('Downloading video files')
    download_video_files_baseball(videos_csv, dataDir, imported)
    print(f'Number of video files: {len(os.listdir(dataDir))}')
    print(f'conf_thresh: {conf_thresh}, confidence_club: {confidence_club}, dataDir: {dataDir}, aiDir: {aiDir}, output: {output}, model: {model}, keyword: {keyword}')
    videoTags = videos_processing(conf_thresh, confidence_club, dataDir, aiDir, output, model, keyword, progress)

    if tag_mode:
        videoIds = pd.read_csv(videos_csv)['videoIds']
        tag_videos_baseball(videoIds, videoTags)

    update_log(run_date, model_name, size, keyword, 'Done')
    
    return 'Done'


def run_pipeline_event(conf_thresh, confidence_club, date_start, date_end, size, model_name, imported, tag_mode, keyword):
    return run_pipeline(conf_thresh, confidence_club, date_start, date_end, size, model_name, imported, tag_mode, keyword, api=False)


if __name__ == "__main__":
    # Run the pipeline
    run_pipeline("./data/")

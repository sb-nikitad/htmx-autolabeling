import subprocess
import os
import json

from inference_3d import get_3d_result


def generate_ai_json(path3d):
    json3d = json.load(open(path3d))
    os.system('java -jar ./tools/sportsbox-tracker-v1.2.0/trackers-1.0-SNAPSHOT-all.jar GENERATE_TRACKER %s tracker_output.json' % json.dumps(json3d[:5], separators=(',', ':')))
    jsonai = json.load(open('tracker_output.json'))
    return jsonai

def save_inter_predictions(videos_path, output_path):
    for video_path in os.listdir(videos_path):
        json2d, json3d = get_3d_result(video_path, hip_distance=121)


def swing_stage_detection(videos_path, json_path):
    swings_stages = dict()

    for video_path in os.listdir(videos_path):
        # json2d, json3d = get_3d_result(os.path.abspath(os.path.join(videos_path, video_path)), hip_distance=121)
        # data = generate_ai_json(os.path.join(json_path, f'{video_path}_3d.json'))
        os.system(f'java -jar /home/gpuuser/HumanPose/quality-pipeline/tools/sportsbox-tracker-v1.2.0/trackers-1.0-SNAPSHOT-all.jar GENERATE_TRACKER_BATCH {json_path} tools/trackers_output')
        
        data = json.load(open(os.path.join('trackers_output.json', f'{video_path}_ai.json')))
    # print(process.returncode)
    
    # for res in os.listdir(output_path):
        # name = res.split('.')[0]
        name = os.path.basename(video_path)
        # with open(res) as f:
        #     data = json.load(f)
        swing = data['swing']
        swings_stages[name] = swing
        # start = swing['start']
        # end = swing['end']
        # swing_stages= swing['position']

    return swings_stages

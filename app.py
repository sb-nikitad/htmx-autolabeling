from flask import Flask, request, render_template, jsonify
import urllib.request
import numpy as np
import os
import requests
from urllib.parse import quote

from tools.autolabeling_pipeline import run_pipeline
from tools.autolabeling import inference_2d_skeleton, box2cs, videos_processing

import cv2

import traceback
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.ERROR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch-videos')
def fetch_videos():
    api_url = "https://us-central1-sportsbox-development.cloudfunctions.net/api/utility/debug"
    response = requests.post(api_url)
    videos = response.json()

    for video in videos:
        video['url'] = quote(video['url'], safe='')

    return render_template('video.html', videos=videos)

@app.route('/process-video')
def process_video():
    video_url = request.args.get('url')
    if video_url:
        video_path = download_video(video_url)
        frame_count = 12
        os.remove(video_path)
        
        return f"<div>Frame count: {frame_count}</div>"
    return "<div>Invalid video URL</div>"

@app.route('/test')
def process_video_test():
  try:
    image = cv2.imread('test.jpg')

    if image is not None:
      print('Image loaded successfully')

    height, width, channels = image.shape

    bbox = [0, 0, width, height] 

    aspect_ratio = 1.0 

    print(bbox)

    preds, maxvals = inference_2d_skeleton(image, "models/lpn_pose2d_tflite_38_177k.tflite", bbox)
    return jsonify({'predictions': preds.tolist(), 'confidences': maxvals.tolist() })
  except Exception as e:
    app.logger.error('An error occurred: %s\n%s', str(e), traceback.format_exc())
    return jsonify({'error': 'An internal error occurred', 'message': str(e)}), 500

@app.route('/test2')
def process_video_test2():
  try:
    dataDir = 'test-vids'
    aiDir = '/mnt/nas01/Uploads/videos/SSL_Videos/Auto_Labelling_Results/AutoLabel_Web/output'
    output_path = 'test-output'
    model = 'models/lpn50_coco2017.tflite'

    confidence = 0.8
    confidence_club = 0.7
    keyword = ''

    run_pipeline(confidence, confidence_club, imported = false, ketyword = keyword, api=True, size=1)
    return "succ"
  except Exception as e:
    app.logger.error('An error occurred: %s\n%s', str(e), traceback.format_exc())
    return jsonify({'error': 'An internal error occurred', 'message': str(e)}), 500

def download_video(url):
    video_stream = urllib.request.urlopen(url)
    video_data = video_stream.read()
    video_filename = 'temp_video.mp4'  
    with open(video_filename, 'wb') as f:
        f.write(video_data)
    return video_filename

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
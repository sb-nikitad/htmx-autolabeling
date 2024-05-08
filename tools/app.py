import gradio as gr
from autolabeling_pipeline import run_pipeline, run_pipeline_event
from autolabeling_pipeline_baseball import run_pipeline as run_pipeline_baseball
from autolabeling_pipeline_baseball import run_pipeline_event as run_pipeline_event_baseball
from fastapi import FastAPI, HTTPException
import uvicorn
import os
import pandas as pd



def process_csv(file, videos_number):
    if file is None:
        return [gr.update(visible=True), gr.update(visible=True), videos_number]
    imported_file = pd.read_csv(file.name)
    videos_number = len(imported_file)
    return [gr.update(visible=False), gr.update(visible=False), videos_number]

CUSTOM_PATH = "/"
LOGS_PATH = "/logs"

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel
from typing import Optional, List


class SwingsData(BaseModel):
    swing_name: list
    user_account: list
    session_name: list
    phone_recorded: list
    fps: list
    pelvis_width_inches: list
    pelvis_width_mm: list
    video_id: list
    video_url: list

class RunAutolabeling(BaseModel):
    confidence: float = 0.8
    confidence_club: float = 0.7
    keyword: str = ""
    SwingsList: SwingsData

@app.post('/api/run_autolabeling')
def run_autolabeling(run: RunAutolabeling):
    confidence = run.confidence
    confidence_club = run.confidence_club
    keyword = run.keyword
    try:
        SwingsList_df = pd.DataFrame(run.SwingsList.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail="SwingsList is not correct")
    # Validate the SwingsList data
    if len(SwingsList_df) == 0:
        # Return bad request HTTP code
        raise HTTPException(status_code=400, detail="SwingsList is empty")
    if len(SwingsList_df.columns) != 9:
        raise HTTPException(status_code=400, detail="SwingsList fields are not correct")
    if not all(x in SwingsList_df.columns for x in ['swing_name', 'user_account', 'session_name', 'phone_recorded', 'fps', 'pelvis_width_inches', 'pelvis_width_mm', 'video_id', 'video_url']):
        raise HTTPException(status_code=400, detail="SwingsList fields are not correct")


    swings_path = '/mnt/nas01/Uploads/videos/SSL_Videos/Auto_Labelling_Results/AutoLabel_Web/SwingsList.csv'
    SwingsList_df.to_csv(swings_path, index=False)
    print(SwingsList_df)
    return run_pipeline(confidence, confidence_club, imported = swings_path, keyword=keyword, api=True, size=len(SwingsList_df))

@app.post('/api/run_autolabeling_baseball')
def run_autolabeling_baseball(run: RunAutolabeling):
    confidence = run.confidence
    confidence_club = run.confidence_club
    keyword = run.keyword
    try:
        SwingsList_df = pd.DataFrame(run.SwingsList.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail="SwingsList is not correct")
    # Validate the SwingsList data
    if len(SwingsList_df) == 0:
        # Return bad request HTTP code
        raise HTTPException(status_code=400, detail="SwingsList is empty")
    if len(SwingsList_df.columns) != 9:
        raise HTTPException(status_code=400, detail="SwingsList fields are not correct")
    if not all(x in SwingsList_df.columns for x in ['swing_name', 'user_account', 'session_name', 'phone_recorded', 'fps', 'pelvis_width_inches', 'pelvis_width_mm', 'video_id', 'video_url']):
        raise HTTPException(status_code=400, detail="SwingsList fields are not correct")


    swings_path = '/mnt/nas01/Uploads/videos/SSL_Videos/Auto_Labelling_Results/AutoLabel_Web/SwingsList.csv'
    SwingsList_df.to_csv(swings_path, index=False)
    print(SwingsList_df)
    return run_pipeline_baseball(confidence, confidence_club, imported = swings_path, keyword=keyword, api=True, size=len(SwingsList_df))

def load_models():
    models = []
    for model in os.listdir('./models'):
        if model.endswith(".tflite") and 'lpn' in model:
            models.append(model.split(".")[0])
        elif model.endswith(".tflite") and 'Sportsbox_Golf_Pose2D' in model:
            models.append(model.split(".")[0])
        # elif model.endswith(".tflite") and 'Baseball' in model:
        #     models.append(model.split(".")[0])
    return models

def load_baseball_models():
    models = []
    for model in os.listdir('./models'):
        if model.endswith(".tflite") and 'Baseball' in model:
            models.append(model.split(".")[0])
    return models

def update_logs():
    print('Updating logs')
    if os.path.exists("/mnt/nas01/Uploads/videos/SSL_Videos/Auto_Labelling_Results/AutoLabel_Web/logs.csv"):
        logs_df = pd.read_csv("/mnt/nas01/Uploads/videos/SSL_Videos/Auto_Labelling_Results/AutoLabel_Web/logs.csv")
    else:
        logs_df = pd.DataFrame(columns=["Date", "Video", "Status", "Message"])
    # Create html table from pandas dataframe
    html_table = logs_df.to_html()
    return html_table


if __name__ == "__main__":
    models = load_models()
    baseball_models = load_baseball_models()
    print(baseball_models)
    with gr.Blocks() as demo:
        with gr.Tab("Autolabeling"):
            with gr.Column():
                gr.Markdown("### Import")
                with gr.Row():
                    import_file = gr.File(label='Import', file_types=['csv'])
                    # export_file = gr.File(label='Export')
                with gr.Row():
                    gr.Examples(
                        examples=['./tools/examples/example.csv'],
                        inputs=import_file,
                    )
                # output_dir = gr.Textbox(label="Output Directory")
                gr.Markdown("### Parameters")

                gr.Markdown('Enter date in YYYY-MM-DD format. E.g. 2023-01-01')
                date_start = gr.Textbox(label="Date Start", placeholder="YYYY-MM-DD", value="2023-01-01")
                date_end = gr.Textbox(label="Date End", placeholder="YYYY-MM-DD", value="2023-02-01")

                videos_number = gr.Slider(label="Number of videos", minimum=1, maximum=850, step=1, value=1, interactive=True) 
                confidence = gr.Slider(label="Confidence threshold", minimum=0, maximum=1, step=0.05, value=0.8)
                
                confidence_club = gr.Slider(label="Confidence threshold club", minimum=0, maximum=1, step=0.05, value=0.7, interactive=True)

                model_selector = gr.Dropdown(models, label="Models", info="Choose model for processing", value=models[2], interactive=True)

                keyword = gr.Textbox(label="Keyword", placeholder="Keyword", value="")

                tag_mode = gr.Checkbox(label="Tag mode", value=False, info="If checked, the pipeline will assign tags to videos")

                submit = gr.Button()
                gr.Markdown('Output folder: /mnt/nas01/Uploads/videos/SSL_Videos/Auto_Labelling_Results/AutoLabel_Web')
                
                estimated_time = gr.Markdown('')
                status = gr.Markdown('')

                import_file.change(fn=process_csv, inputs=[import_file, videos_number], outputs=[date_start, date_end, videos_number])

                # submit.click(fn=run_pipeline, inputs=[import_file, confidence], outputs=[status])
                submit.click(fn=lambda size: estimated_time.update(f"Estimated processing time: {2*size} minutes"), inputs=[videos_number], outputs=[estimated_time])
                submit.click(fn=run_pipeline_event, inputs=[confidence, confidence_club, date_start, date_end, videos_number, model_selector, import_file, tag_mode, keyword], outputs=[status], api_name="run_autolabeling")
                
        # Baseball 
        with gr.Tab("Autolabeling Baseball"):
            with gr.Column():
                gr.Markdown("### Import")
                with gr.Row():
                    import_file = gr.File(label='Import', file_types=['csv'])
                    # export_file = gr.File(label='Export')
                with gr.Row():
                    gr.Examples(
                        examples=['./tools/examples/example.csv'],
                        inputs=import_file,
                    )
                # output_dir = gr.Textbox(label="Output Directory")
                gr.Markdown("### Parameters")

                gr.Markdown('Enter date in YYYY-MM-DD format. E.g. 2023-01-01')
                date_start = gr.Textbox(label="Date Start", placeholder="YYYY-MM-DD", value="2023-01-01")
                date_end = gr.Textbox(label="Date End", placeholder="YYYY-MM-DD", value="2023-02-01")

                videos_number = gr.Slider(label="Number of videos", minimum=1, maximum=850, step=1, value=1, interactive=True) 
                confidence = gr.Slider(label="Confidence threshold", minimum=0, maximum=1, step=0.05, value=0.8)
                
                confidence_club = gr.Slider(label="Confidence threshold club", minimum=0, maximum=1, step=0.05, value=0.7, interactive=True)

                model_selector = gr.Dropdown(baseball_models, label="Models", info="Choose model for processing", value=baseball_models[0], interactive=True)

                keyword = gr.Textbox(label="Keyword", placeholder="Keyword", value="")

                tag_mode = gr.Checkbox(label="Tag mode", value=False, info="If checked, the pipeline will assign tags to videos")

                submit = gr.Button()
                gr.Markdown('Output folder: /mnt/nas01/Uploads/videos/SSL_Videos/Auto_Labelling_Results/AutoLabel_Web')
                
                estimated_time = gr.Markdown('')
                status = gr.Markdown('')

                import_file.change(fn=process_csv, inputs=[import_file, videos_number], outputs=[date_start, date_end, videos_number])

                # submit.click(fn=run_pipeline, inputs=[import_file, confidence], outputs=[status])
                submit.click(fn=lambda size: estimated_time.update(f"Estimated processing time: {2*size} minutes"), inputs=[videos_number], outputs=[estimated_time])
                submit.click(fn=run_pipeline_event_baseball, inputs=[confidence, confidence_club, date_start, date_end, videos_number, model_selector, import_file, tag_mode, keyword], outputs=[status], api_name="run_autolabeling_baseball")
                
        with gr.Tab("Logs") as logs:
            with gr.Column():
                gr.Markdown("### Logs")
                html_component = gr.HTML(value=update_logs, every=10)

    demo.queue(concurrency_count=1)
    app = gr.mount_gradio_app(app, demo, path=CUSTOM_PATH)
    uvicorn.run(app, host="0.0.0.0", port=40040, log_level="debug")
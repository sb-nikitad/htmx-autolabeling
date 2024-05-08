import os
import pandas as pd

def log_run(date, model, videos_number, keyword, status):
    if os.path.exists("/mnt/nas01/Uploads/videos/SSL_Videos/Auto_Labelling_Results/AutoLabel_Web/logs.csv"): 
        log_file = pd.read_csv("/mnt/nas01/Uploads/videos/SSL_Videos/Auto_Labelling_Results/AutoLabel_Web/logs.csv")
    else:
        log_file = pd.DataFrame(columns=['id', 'model', 'videos_number', 'keyword', 'status'])
    print(log_file)
    id =  date + '_'+ model + '_' + str(videos_number) + '_' + keyword
    log_file = log_file.append({'id': id, 'model': model, 'videos_number': videos_number, 'keyword': keyword, 'status': status}, ignore_index=True)
    log_file.to_csv("/mnt/nas01/Uploads/videos/SSL_Videos/Auto_Labelling_Results/AutoLabel_Web/logs.csv", index=False)

def update_log(date, model, videos_number, keyword, status):
    log_file = pd.read_csv("/mnt/nas01/Uploads/videos/SSL_Videos/Auto_Labelling_Results/AutoLabel_Web/logs.csv")
    id = date + '_' + model + '_' + str(videos_number) + '_' + keyword
    log_file.loc[log_file['id'] == id, 'status'] = status
    log_file.to_csv("/mnt/nas01/Uploads/videos/SSL_Videos/Auto_Labelling_Results/AutoLabel_Web/logs.csv", index=False)

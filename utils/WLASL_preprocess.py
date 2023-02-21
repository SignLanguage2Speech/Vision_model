import os
import json
import subprocess
import pandas as pd
import numpy as np
FPS = 25 # default for all WLASL files

wlasl_wd = '/work3/s204503/bach-data/WLASL'

def convertDataFormat(filename='WLASL_v0.3.json', save=False):
    # os.chdir(os.path.join(os.getcwd(), 'data/WLASL')) # original
    os.chdir(os.path.join(wlasl_wd))   # path to .json file | for HPC scratch (s204503)
    file = open(filename)
    data = json.load(file)
    os.chdir(os.path.join(wlasl_wd, 'WLASL2000'))   # path to .mp4-files | for HPC scratch (s204503)
    data_dict = {'gloss': [],
                 'label':[],
                 'video_id': [],
                 'split':[],
                 'frame_start': [],
                 'frame_end': [],
                 'fps':[]}  
           
    # get relevant data         
    for dictionary in data:
        for obs in dictionary['instances']:
            data_dict['gloss'].append(dictionary['gloss'])
            data_dict['video_id'].append(obs['video_id'])
            data_dict['split'].append(obs['split'])
            data_dict['frame_start'].append(obs['frame_start'])
            data_dict['frame_end'].append(obs['frame_end'])
            data_dict['fps'].append(obs['fps'])

    # get labels
    words = data_dict['gloss']
    words = list(sorted(set(words)))
    for i in range(len(data_dict['gloss'])):
        data_dict['label'].append(words.index(data_dict['gloss'][i]))
        
    df = pd.DataFrame.from_dict(data_dict)
    if save:
        df.to_csv('WLASL_labels.csv')

    return df

def video2jpg(vname, input_dir, output_dir, fps=FPS, start=1, end=-1, lower_quality=False):
    if '.mp4' not in vname:
        return
    
    name, ext = os.path.splitext(vname)
    input_path = os.path.join(input_dir, vname)
    output_path = os.path.join(output_dir, name)

    try:
        if os.path.exists(output_path):
            if not os.path.exists(os.path.join(output_path, 'image_00001.jpg')):
                subprocess.call(f'del {output_path}', shell=True)
                print(f"removed existing instance of {output_path} and created new empty folder...")
                #os.mkdir(output_path)
            else:
                print(f"Video at {output_path} has already been converted")
        else:
            os.mkdir(output_path)
    
    except Exception as e:
        print(e)
        print(output_path)
        return

    # if gloss is contained in entire video..
    if start == 1 and end == -1:
        if not lower_quality:
            cmd = f'ffmpeg -i {input_path} -threads 1 -r {fps} -vf scale=-1:331 -qscale:v 0 {output_path}/img_%05d.jpg'
        else:
            cmd = f'ffmpeg -i {input_path} -threads 1 -r {fps} -vf scale=-1:331 {output_path}/img_%05d.jpg'
    else:
        print("Cropping not implemented because labels seem wrong....")
    # call cmd
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    # create dataframe of labels, ids, etc...
    df = convertDataFormat(save=True)
    #print(df.head(5))

    # get images from videos...
    #os.chdir('C:/Users/micha/OneDrive/Skrivebord/Vision_model/data/WLASL')
    # input_dir = 'WLASL_videos'
    input_dir = '' # HPC
    #output_dir = 'WLASL_images'
    # video_names = os.listdir(os.path.join(os.getcwd(), input_dir)) # original
    video_names = os.listdir(os.path.join(wlasl_wd, input_dir))   # for HPC scratch (s204503)
    df_train = df.loc[df['split'] == 'train']
    

    # get images from videos...
    #os.chdir('C:/Users/micha/OneDrive/Skrivebord/Vision_model/data/WLASL')
    #input_dir = 'WLASL_videos'
    #output_dir = 'WLASL_images'
    #video_names = os.listdir(os.path.join(os.getcwd(), input_dir))
    #print("converting videos to images...")
    #for name in video_names:
    #   video2jpg(name, input_dir, output_dir)
    
    ### find and show what videos are missing in the df that are in the videos folder...
    #vids = list(df['video_id'])
    #vids = [e+'.mp4' for e in vids]
    #for name in video_names:
    #    if name not in vids:
    #        print(name)
    

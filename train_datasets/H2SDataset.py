import os
import numpy as np
import subprocess

val_path = '/work3/s204503/bach-data/how2sign/val/raw_videos'

def get_subvideos(video_name, df):
    
    return None

def video2array(vname, input_dir=os.path.join(os.getcwd(), 'data/WLASL/WLASL_videos'), fps=25):
  H = W = 256 # default dims for WLASL
  name, ext = os.path.splitext(vname)
  video_path = os.path.join(input_dir, vname)
  # out = []
  get_no_of_frames = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', video_path]
  n_frames = int(subprocess.check_output(get_no_of_frames))
  # pdb.set_trace()
  out = np.zeros((n_frames,H,W,3))
  cmd = f'ffmpeg -i {video_path} -f rawvideo -pix_fmt rgb24 -threads 1 -r {fps} pipe:'
  pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, bufsize=10**8, stderr=subprocess.DEVNULL)

  # while True:
  for i in range(n_frames):
    buffer = pipe.stdout.read(H*W*3)
    if len(buffer) != H*W*3:
      break
    # out.append(np.frombuffer(buffer, np.uint8).reshape(H, W, 3))
    out[i,:,:,:] = np.frombuffer(buffer, dtype=np.uint8).reshape(H, W, 3)
    # pdb.set_trace()
  pipe.stdout.close()
  pipe.wait()
  # pdb.set_trace()
  return out


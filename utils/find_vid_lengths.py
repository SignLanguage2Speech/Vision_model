import os
import numpy as np
import subprocess 

input_dir = os.path.join(os.getcwd(), 'data\\WLASL\\WLASL_videos')
vnames = os.listdir(input_dir)

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

lengths = []
for i, video in enumerate(vnames):
    lengths.append(get_length(os.path.join(input_dir, video)))
    if i % 100 == 0:
        print(f"at video {i}/{len(vnames)}")

lengths = sorted(lengths)
print(max(lengths))
print(np.mean(lengths))
print(lengths[len(lengths)-30:])
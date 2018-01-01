from moviepy.editor import *

clip1 = VideoFileClip("../CarND-Vehicles-Detection-P5/out_video_project2.mp4").subclip(36,40)
clip1.write_gif("final.gif")
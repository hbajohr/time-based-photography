#!/usr/bin/python
# coding: utf-8
import subprocess as sp
import cv2
import os


def film_data(input_file):
	file_path = input_file 
	vid = cv2.VideoCapture(file_path)
	video_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
	video_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
	property_id = int(cv2.CAP_PROP_FRAME_COUNT)
	length = int(cv2.VideoCapture.get(vid, property_id))	
	return (video_height, video_width, length)
		
def makefilm(directory, output,fps): # to do: make output video - had a problem with ffmpeg here
	'''print "MAKE VIDEO...."
	command = ["ffmpeg -pattern_type glob -i " + directory + "/*.jpg -c:v libx264 -r 30 "+ output + ".mp4"]
	cmd = sp.Popen(command, shell=True,stdout=sp.PIPE)
	cmd.stdout.flush()
	'''
	image_folder = directory
	video_name = output + ".mp4"
	images = [img for img in sorted(os.listdir(image_folder), reverse = False) if img.endswith(".jpg")]
	frame = cv2.imread(os.path.join(image_folder, images[0]))
	height, width, layers = frame.shape
	fourcc = cv2.VideoWriter_fourcc(*'MPEG')
	video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))
	for image in reversed(images):
		video.write(cv2.imread(os.path.join(image_folder, image)))
	cv2.destroyAllWindows()
	video.release()
	print("VIDEO DONE")
	
def remove_files(directory):
	command = ["rm "+ directory +"/pan_img-*.jpg"]
	cmd = sp.Popen(command, shell=True,stdout=sp.PIPE)
	cmd.stdout.flush()
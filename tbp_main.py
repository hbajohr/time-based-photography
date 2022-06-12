# -*- coding: utf-8 -*-

import subprocess as sp
import numpy 
from PIL import Image
import os
import sys
import time
import datetime
from tbp_lib import * # this needs to be in the same directory
start_time = time.process_time()

#### basic variables - change here manually
project_name = "PROJECT NAME" 
input_file = "/Users/tbp/text.mov"
directory = "/Users/tbp/" + project_name

start_frame = 1 	# first frame to start from
slice_width = 1		# width of the "slices"
frame_steps = 1 	# steps between the frames
pixel_steps = 1 	# steps between slices (doesn't work yet)
start_time = 0 		# start time
end_time = 0 		# end time

#### start
print("")
print("Time-based photography script for Python, Hannes Bajohr 2015-2022")
print("-----------------------------------------------------------------")
print("Panorama: " + project_name)
sys.stdout.write('Starting at frame {} with a slice width of {} (the {}th image after each {}th frame) \n\n'.format(start_frame, slice_width, pixel_steps, frame_steps))

#### extract info from the video file
print("++ Reading video file info")
video_height = int(film_data(input_file)[0]) # to do: refactor
video_width = int(film_data(input_file)[1])
frame_count = int(film_data(input_file)[2])
print("		Height:  " + str(video_height))
print("		Width: " + str(video_width))

print("++ Reading frame count")
image_width = int(frame_count)*slice_width
print("		Frame count: " + str(frame_count))
print("		Image width: " + str(image_width))

print("++ Processing")
output_image = Image.new("RGB", (image_width,video_height)) 

for j in range(0,frame_count):	# each pass = one image
	if j % frame_steps == 0:
		
		sys.stdout.write('		FRAME {}/{} - PROGRESS {}% – REMAINING {}\r'.format(str(j), str(frame_count), round(float(j)/float(frame_count)*100), str(datetime.timedelta(seconds=round((end_time-start_time)*((frame_count-j)/frame_steps)))))) ##ÄNDERN - ES IST NICHT DER FRAMECOUNT SONDERN DIE BILDBREITE - DAS SCRIPT KANN NUR DIE STREIFEN DER BILDBREITE DURCHLAUFEN; DANAHC WIRD ES SCHWARZ
		sys.stdout.flush()
		start_time = time.time()
		command = [ 'ffmpeg',
			'-i', input_file, 
			'-f', 'image2pipe',
			'-pix_fmt', 'rgb24',
			'-vcodec', 'rawvideo','-', '-nostats', '-hide_banner', '-loglevel', '0']
		pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
		pipe.stdout.flush()
		
		for i in range(0,image_width,pixel_steps): # each pass = one "slice"
			try:
				
				raw_image = pipe.stdout.read(video_height*video_width*3) 
				image =  numpy.frombuffer(raw_image, dtype='uint8')
				#image = image.reshape((BREITEfilm,HOEHEfilm,3)) # toggle between this and the next line if output is crap
				image = image.reshape((video_height,video_width,3)) 
				im = Image.fromarray(image)
				#im = im.rotate(-90) # toggle for rotation (to do: automate)
				w,h = im.size
				im = im.crop((start_frame, 0, w-(w-slice_width)+start_frame, h))
				output_image.paste(im,(i*slice_width,0))
			except:
				print("error at %s" % (i))
				i=0
				break

		if not os.path.exists(directory):
			os.makedirs(directory)
		output_image.save(directory + "/pan_img-%04d.jpg" % (j))
	end_time=time.time()
	start_frame=start_frame+1
	
# MAKE FILM 
makefilm(directory, directory + "/" + project_name,30)
tprint("This took" + time.process_time() - start_time + " seconds.")


# to do: ask to delete temporary images
'''reply = str(input("++ Delete temporary images? ")) 
if reply == "j":
	remove_files(directory)		
	print("		Images removed")
	print("++DONE")
else:
	print("++DONE")'''
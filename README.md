# Time-based photography 

## Installation:

Simply copy both files tbp_main.py and tbp_lib.py into a folder; open the main file in an editor, add the path info, and change the parameters. 
I might make a version with flags later, but I like to see the possible options. You will need to install the dependencies numpy and PIL. 

## Description:


This is a project I have been working on and off for more than a decade (the earliest version I have on my drive is from 2008 and was started in Processing, but I let it rest for quite a while). The basic idea is to conceive of a photograph as a _function over the time of a video file_. While a photograph is a projection of a point in time onto a static image, time based photography, as I understand it, compresses a temporal sequence into such a static image. 

The project works in two ways: 

## 1) Video -> Image(time)
For each frame of a video file, a single vertical strip of a pretermined width is extracted; the sum of these strips is concatenated and returned as an image file. 

### 1.1) Static camera, moving objects
Here is an example of a person walking past my window. 

**Input video (recorded in slow motion): **

https://user-images.githubusercontent.com/20578427/173235913-1f1f20f8-a49e-4cd6-aca5-415c1f03d600.mp4

**Output image: **

![time based photography](https://hannesbajohr.de/IMGS/fenster2.jpg)

The output input is a function of time of the input video; as the video is static, only the differential movements are captured in the image. 
Notice that this translates to interestingly distortions where some parts are longer in one place than others (the feet), and to a reversal of directions: Even though the car is passing by from right to left in the video, in the image, it appears to drive from left to right. The same with the people, who also walk in different directions but in the image appear to face the same way. The reason for this is that all movements are mapped continuously onto the image. This video helps to understand how it works: 

https://user-images.githubusercontent.com/20578427/173237498-fb535e47-5223-460f-879b-75049085b4eb.mp4

Here, it is best to use slow-motion footage so that ideally each frame of the output image covers the movement of one pixel in the video. The movement mapping and its distortion are especially interesting for moving bodies. This is, for example, how I made my profile image: 

**Input video:**

https://user-images.githubusercontent.com/20578427/173238005-a3de95e9-b58d-426a-a761-178c578ad921.mov

**Output image:**

![line-0001aFrame1](https://user-images.githubusercontent.com/20578427/173240731-1a962231-fcdb-4b2d-b460-d49d4642036d.jpg)


(The video was longer, I turned more than once. Notice how I sometimes stray too far from my axis so that, for instance, my mouth appears twice, hanging in mid-air.)

### 1.1) Moving camera, static objects

Another approach is to move the _camera_ so as to avoid the background repetition. Here, the effect is similar to the "panorama" feature on many cell phones. Since here we don't have a single vanishing point, this is also the reduction of _perspectival_ space in a static image, in effect producing an isomorphic image (or at least an image with more than one vanishing point).

Here is a video of my old stop at 125th St in New York turned into an image. (This time, the "slices" are wider than one pixel as this was taken in, I think, 2013 before the iPhone allowed slo-mo.)

**Input video:**

https://user-images.githubusercontent.com/20578427/173238427-b783d30c-9ef1-47f5-a569-d2c27184f352.mov

**Output video:**

<img alt="125th" src="https://user-images.githubusercontent.com/20578427/173238562-55f4433b-209d-4aee-82ab-272db4e39354.png">

Here is another, smoother image from a slo-mo video. 

**Input video:**

https://user-images.githubusercontent.com/20578427/173238814-a722ebbb-831f-48a1-8fd9-63e41ee52768.mp4

**Output video:**

![trainride](https://user-images.githubusercontent.com/20578427/173238901-e32931ff-d438-41ea-bc01-8f884b53518e.png)

You can see how objects moving in the foreground move faster and are thus compressed, while those in the background, moving slower, are extended. Notice also how the straight bridge becomes curved due to the differential movement. (Panofsky would have liked this.) 

##2) Image(time) -> Video(perspective): Operation 1) is repeated for _every possible vertical "slice"._ 

For each of the images above, I had to make the decision _which pixel on the x axis_ I chose as the position of the "slice." For instance, for the very first image, I chose the center slice (the vertical line equidistant from each of first and the last row of pixels in the image; you can see this in the composite video well, where I had to cut the video in half to show how it maps onto the image). 

But for an image of a width of _n_ pixels, there are also _n_ possible positions for these "slices." The output will be different as time passes between the first and the last, be it the camera or an object; also, especially for videos with a lot of depth, there is a slight difference in perspective between the first and the last "slice."

For example, this is the output for the video from above of the 125th St stop, taken at difference "slice" positions (the video was 1080 pixel wide):

<img width="300" alt="first row" src="https://user-images.githubusercontent.com/20578427/173253561-42a6edf0-e1ed-47c3-ae7d-6204b6ba638d.png">

Very first row of pixels (_n_=0):

<img width="1684" alt="first row" src="https://user-images.githubusercontent.com/20578427/173239476-0ce854c1-2847-42b6-a7b8-852d7b4b95a1.png">

Center row of pixels (_n_=540): 

<img width="1686" alt="middle row" src="https://user-images.githubusercontent.com/20578427/173239469-1a7cacdc-f08a-4305-a4bd-79b2cbcdc5a4.png">

Last row of pixels (_n_=1080):

<img width="1685" alt="last row" src="https://user-images.githubusercontent.com/20578427/173239462-8a1be82b-ac22-43ef-a75e-35233464b3d1.png">

This makes intuitive sense of you remember that this is taken from a moving train: The corner of the building in the middle would move in and out of frame, first only as an edge blocking the view into the street, then lining up with the perspective of the street, and finally showing the facades of the street. Since the last row of pixels captures the first phase and the first row (as the camera is moving from left to right) the last phase of that movement (time having passed), the perspectives are different. 

Now, since this can be done for all _n_ rows of pixels of a video, one can turn the total of all these slightly different images _into a video again_. The result is something like a derivative of the video's time function. 

Here it is for the 125th St stop (still a bit blocky): 

https://user-images.githubusercontent.com/20578427/173239850-62794041-c412-45a8-967b-d89da8790b8f.mp4

(This was an early attempt; I made a mistake in the script so the last few slices are repeated.)

And here is the "derivative" for the other train video: 

https://user-images.githubusercontent.com/20578427/173240020-b6385512-9c54-4c99-994d-4d733f5fbb35.mp4

I like how the fact that the background in the video moves slower than the foreground translates into the reverse: Now it is the background that moves while the foreground remains static - it is as if the perspective had been flipped in some way I don't quite have a name for yet. 

This works also for videos where the camera remains static. Here is a similar shot of my old street as above: 

https://user-images.githubusercontent.com/20578427/173240540-45254d1c-adc7-4152-a2aa-1f28b265d6c3.mp4

Two things are interesting here: First, the slow black flicker on the bottom half of the screen are the vertical bars of the fence you see in the first video. And second, you can see how the people in the background appear to go backwards - they are still _flipped_ but now move in the correct direction!

And here is a similar shot as before with me rotating: 

https://user-images.githubusercontent.com/20578427/173242039-155ca2d0-e168-4b77-9fb9-eb4692da32c8.mp4

## Combining static and moving approach

Recently, I have been combining the static and the moving approach by mounting my phone on a slow rotating tripod. This way, you get both the background as well as moving objects without either of them blurring. The results can be absolutely freaky, as in this street scene from Berlin: 

**Input video:**


https://user-images.githubusercontent.com/20578427/173242633-a52d55fe-c7ac-43e9-a97e-6ebf3bf8c7e7.mp4

**Output video: **

https://user-images.githubusercontent.com/20578427/173243806-0a2f4e4c-f637-423f-b8ba-ee97a93aa18b.mp4


To me, this looks insane - the combination of both approaches makes them look like they are floating, and the perspectival stretching/compressing makes the jogger smaller than the lamppost he is jogging in front of. 



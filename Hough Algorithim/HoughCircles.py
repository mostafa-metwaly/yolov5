# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 20:20:24 2021

@author: AWasf
"""

# all plots will be set directly below the code cell that produced it
# %matplotlib inline
from matplotlib import pyplot as plt
# from IPython.display import clear_output
# set inline plots size
plt.rcParams["figure.figsize"] = (16, 10) # (w, h)
# remove grid lines
import numpy as np
import time
import cv2

"""
        HELPER FUNCTIONS
"""
# funcrion to read and resize an image
def read_and_resize(filename, grayscale = False, fx= 0.5, fy=0.5):
    if grayscale:
      img_result = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
      imgbgr = cv2.imread(filename, cv2.IMREAD_COLOR)
      # convert to rgb
      img_result = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2RGB)
    # resize
    img_result = cv2.resize(img_result, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    return img_result

    
def showInRow(list_of_images, titles = None, disable_ticks = False):
  plt.rcParams["figure.figsize"] = (16, 10) # (w, h)
  count = len(list_of_images)
  for idx in range(count):
    subplot = plt.subplot(1, count, idx+1)
    if titles is not None:
      subplot.set_title(titles[idx])
      
    img = list_of_images[idx]
    cmap = 'gray' if (len(img.shape) == 2 or img.shape[2] == 1) else None
    subplot.imshow(img, cmap=cmap)
    #subplot.imshow(img, cmap=cmap,vmin=0, vmax=1) to disable normalization
    if disable_ticks:
      plt.xticks([]), plt.yticks([])
  plt.show()

 # function for colors array generation
def generate_colors(num):
  r = lambda: np.random.randint(0,255)
  return [(r(),r(),r()) for _ in range(num)] 

def process_video(video_path, frame_process):
  vid = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
  try:
    while(True):
      ret, frame = vid.read()
      if not ret:
        vid.release()
        break

#       frame = cv2.resize(frame, (int(frame.shape[1]), int(frame.shape[0])))        
#       frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      if frame_process is not None:
        # frame = cv2.flip(frame,1)
        frame = frame_process(frame)
#         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         out.write(frame)
        cv2.imshow("Video",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
            
        if cv2.waitKey(1) & 0xFF == ord('c'): 
            cv2.imwrite("DetectedCylinder.jpg",frame)            

    vid.release()
    cv2.destroyAllWindows()
  
  except KeyboardInterrupt:
    vid.release()
    cv2.destroyAllWindows()
    
    
"""
        Trying out cv2.HoughCircles to detect the circle face of the cylinder 
"""
def detectCircles(image):
    output = image.copy()
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     G_gray = cv2.GaussianBlur(gray,(11,11),0)
#     M_gray = cv2.medianBlur(G_gray,9)
#     showInRow([binary])  
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 500, param1=200, param2=50, maxRadius=50)     # 1.5, 100, param1=200, param2=50
#     print(circles.shape)
    # ensure at least some circles were found
    if circles is not None:
      # convert the (x, y) coordinates and radius of the circles to integers
      circles = np.round(circles[0, :]).astype("int")
      # loop over the (x, y) coordinates and radius of the circles
      for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        # z = 220     # z distance from image plane
        z = (40 * f) / (2*r)
        [X,Y,Z] = np.dot(Inv_intrinsic_Mtx, z*np.array([[x],[y],[1]]))
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        cv2.putText(output,"{:.2f},{:.2f},{:.2f}".format(X[0],Y[0],Z[0]), (x+r,y) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)       # show the output image
      # showInRow([cyl1, output])
#       showInRow([output])
#       cv2.waitKey(0)

    return output

"""
        Trying out cv2.HoughCircles to detect the circle face of the cylinder ☼
"""
intrinsic_Mtx = np.load("mtx_logiRobot.npy")                 # Intrinsics Matrix of the Webcam of My Laptop obtained from Calibration
Inv_intrinsic_Mtx = np.linalg.inv(intrinsic_Mtx)   # Computing the Inverse Matrix of the Intrinsics Matrix
f = sum([intrinsic_Mtx[0,0],intrinsic_Mtx[1,1]])/2
process_video(2, detectCircles)
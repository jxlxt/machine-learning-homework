import os
import cv2

path = 'pig'
# file location
files = os.listdir(path)
# get all the file name under the document
s = []
iter = 1
for file in files:
    vidcap = cv2.VideoCapture(path+"/"+file)  # read the video file
    success, image = vidcap.read()
    count = 1  # file number
    timeF = 1000  # Save an image every 50 frames
    num = 1  # frame number
    while success:
        success, image = vidcap.read()
        # print('Capture the %d frame' % count)
        if count % timeF == 0:
            cv2.imwrite("%d_pig_frame%d.jpg" % (iter, num), image)  # write the frame picture into new file
            num += 1
        count += 1
    iter +=1
    print('Finish capturing %s' % file)

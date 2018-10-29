import cv2


vidcap = cv2.VideoCapture('1.mp4') # read the video file
success,image = vidcap.read()
count = 0
timeF = 50  # Save an image every 50 frames
num = 1
while success:
    success, image = vidcap.read()
    # print('Capture the %d frame' % count)
    if count % timeF == 0:
        cv2.imwrite("1_pig_frame%d.jpg" % num, image) # write the frame picture into new file
        num += 1
    count += 1
print('Finish capturing 1!')






from skimage.measure import compare_ssim
import numpy as np
import imutils
import cv2
import pandas as pd

huarray=[]

def image_resize(path, width = None, height = None, inter = cv2.INTER_AREA):
    image=cv2.imread(path,0)
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def writefile(data,path,column):
    df=pd.DataFrame(np.array(data,dtype="object"),columns=column)
    with open(path,'w+') as f:
        df.to_csv(f,mode='w',header=False)

def createMEIsandMHIs(path):
    cap=cv2.VideoCapture(path)
    firstFrame=None
    width,height=cap.get(3),cap.get(4)
    image1 = np.zeros((int(height), int(width)), np.uint8)
    image2 = np.zeros((int(height), int(width)), np.uint8)
    ctr=1
    while True:
        ret,frame=cap.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if firstFrame is None:
            firstFrame = gray
            continue
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        image1=cv2.add(image1,thresh)
        image2=cv2.addWeighted(image2,1,thresh,ctr/1000,0)
        ctr+=1
    cv2.imwrite("mei/MEI.jpg",image1)
    test=image_resize("mei/MEI.jpg",height=640,width=480)
    cv2.imwrite("static/test.jpg",test)
    cv2.imwrite("mei/MHI.jpg",image2)
    image1=image_resize("mei/MEI.jpg",height=120,width=90)
    cv2.imwrite("mei/test.jpg",image1)
    cv2.imwrite("static/test.jpg",image1)
    cap.release()
    cv2.destroyAllWindows()

# import the necessary packages
import argparse
import warnings
import datetime
import json
import time
import sys
try:
  import imutils
  from imutils.video import VideoStream
  import cv2
except Exception as e:
  print("imutils or cv2 not found. Install with python -m pip install imutils opencv-python")
  print("Exception: {}".format(e))
  sys.exit(-1)


def main(args, conf):
    if args.get("video", None) is None:
        vs = VideoStream(src=0).start()
        time.sleep(conf["camera_warmup_time"])
    else:
        vs = cv2.VideoCapture(args["video"])
        
    avg = None
    
    while True:
        #time.sleep(1)
        frame = vs.read()
        frame = frame if args.get("video", None) is None else frame[1]
        text = "Unoccupied"
        
        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        # if the average frame is None, initialize it
        if avg is None:
            avg = blur.copy().astype("float")
            continue
        # accumulate the weighted average between the current frame and previous frames
        cv2.accumulateWeighted(blur, avg, 0.03)
        # Compute the difference between the current frame and running average
        frameDelta = cv2.absdiff(blur, cv2.convertScaleAbs(avg))
        
        # threshold the delta image
        thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes
        dilate = cv2.dilate(thresh, None, iterations=2)
        # find contours on thresholded image
        cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < conf["min_area"]:
                continue
            # compute the bounding box for the contour, draw it on the frame, and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied"
        # draw the text and timestamp on the frame
        cv2.putText(frame, 
                    "Room Status: {}".format(text),
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 255), 
                    2)
        cv2.putText(frame,
                    datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), 
                    (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, 
                    (0, 0, 255), 
                    1)
      
        # check to see if the frames should be displayed to screen
        if conf["show_video"]:
            cv2.imshow("Security Feed", frame)
            #cv2.imshow("Gray", gray)
            cv2.imshow("Average", cv2.convertScaleAbs(avg))
            cv2.imshow("Blur", blur)
            cv2.imshow("Frame Delta", frameDelta)
            cv2.imshow("Thresh", thresh)
            cv2.imshow("Dilated", dilate)
            # if the `q` key is pressed, break from the lop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
if __name__=="__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--conf", required=True, help="path to the JSON configuration file")
    ap.add_argument("-v", "--video", help="path to the video file")
    args = vars(ap.parse_args())
    warnings.filterwarnings("ignore")
    conf = json.load(open(args["conf"]))
    
    main(args, conf)

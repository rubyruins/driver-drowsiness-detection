# USAGE
# python dds.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python dds.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from tkinter import *
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2


root = Tk()
root.geometry('400x300')
root.configure(bg="black")
root.maxsize(400,300)
root.title('Drowsiness Detection System')


def sound_alarm():
	# play an alarm sound
	playsound.playsound("alarm.wav")

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def start(EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES):


	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-w", "--webcam", type=int, default=0,
		help="index of webcam on system")
	args = vars(ap.parse_args())

	# define two constants, one for the eye aspect ratio to indicate
	# blink and then a second constant for the number of consecutive
	# frames the eye must be below the threshold for to set off the
	# alarm 
	

	# initialize the frame counter as well as a boolean used to
	# indicate if the alarm is going off
	COUNTER = 0
	ALARM_ON = False

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	# grab the indexes of the facial landmarks for the left and
	# right eye, respectively
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	# start the video stream thread
	print("[INFO] starting video stream thread...")
	vs = VideoStream(src=args["webcam"]).start()
	time.sleep(1.0)

	i = 0
	min_ear = 100
	max_ear = 0
	ear = 0
	# loop over frames from the video stream
	global text
	while True:
		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()
		frame = imutils.resize(frame, width=650)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		cv2.putText(frame, f"EYE_AR_THRESH = {EYE_AR_THRESH}", (10, 480),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		cv2.putText(frame, f"EYE_AR_CONSEC_FRAMES = {EYE_AR_CONSEC_FRAMES}", (300, 480),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

		# detect faces in the grayscale frame
		rects = detector(gray, 0)

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0

			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				COUNTER += 1

				# if the eyes were closed for a sufficient number of
				# then sound the alarm
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					# if the alarm is not on, turn it on
					if not ALARM_ON:
						ALARM_ON = True

						# check to see if an alarm file was supplied,
						# and if so, start a thread to have the alarm
						# sound played in the background
						t = Thread(target=sound_alarm)
						t.deamon = True
						t.start()

					# draw an alarm on the frame
					cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


			# otherwise, the eye aspect ratio is not below the blink
			# threshold, so reset the counter and alarm
			else:	
				COUNTER = 0
				ALARM_ON = False

			# draw the computed eye aspect ratio on the frame to help
			# with debugging and setting the correct eye aspect ratio
			# thresholds and frame counters
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		if i<50:
			if ear < min_ear:
				min_ear = ear
			elif ear > max_ear:
				max_ear = ear
		elif i == 50:
			EYE_AR_THRESH = (min_ear + max_ear)/2
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()



def settings():

	settingsBtn.pack_forget()
	startBtn.pack_forget()

	
	EATList = ['0.27','0.28','0.29','0.30','0.31','0.32','0.33','0.34']
	EACList = [44,45,46,47,48,49,50,51,52]
	v1 = StringVar()
	v1.set(0.31)
	v2 = IntVar()
	v2.set(48)

	l1 = Label(root, text="Set Threshold EAR: ", bg="black", fg="white")
	l1.pack(pady=5)
	opt1 = OptionMenu(root, v1, *EATList)
	opt1.config(width=90, font=('Helvetica', 12))
	opt1.pack()

	l2 = Label(root, text="Set consecutive frames: ", bg="black", fg="white")
	l2.pack(pady=5)
	opt2 = OptionMenu(root, v2, *EACList)
	opt2.config(width=90, font=('Helvetica', 12))
	opt2.pack()


	ok = Button(root, text="SAVE AND START", command=lambda:start(float(v1.get()), v2.get()), bg="black", fg="white")
	ok.pack(pady=35)


settingsBtn =  Button(root, text="SETTINGS", command=settings, bg="black", fg="white")
settingsBtn.pack(pady=55)

startBtn = Button(root, text="START", command=lambda:start(0.31,48), bg="black", fg="white")
startBtn.pack(pady=35)

root.mainloop()
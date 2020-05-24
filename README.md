
Driver Drowsiness Detection :oncoming_automobile:
============

![made-with-python](https://img.shields.io/badge/made%20with-Python-blue.svg)

This system monitors drowsiness by detecting the driver's face and sending a drowsiness alert if his eyes have been closed for a certain number of frames. 
The user can configure their own threshold by specifying an EAR ratio and customising the number of frames if they do not want to use the defaults. 

---

## Libraries used:
- `dlib` for the face detection model
- `opencv-python` for drawing frames around the eyes
- `tkinter` for the GUI
- `playsound` to play the alarm
- `scipy` for Euclidean distance calculations


---

## Installation:

**You must have Python 3.6 or higher to run the file.**

- Create a new virtual environment for running the application. You can follow the instructions [here.](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/)
- Navigate to the virtual environment and activate it.
- Install the dependancies using `pip install -r requirements.txt`
- Run the `dds.py` file with `python dds.py`

---

This system was made as part of our Mini Project for Semester IV. 

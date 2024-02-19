# Video_Captioning_Project

## Description
Uses Speech Recognition and CV2 to caption videos, outputting a separate video with captions. Also outputs the video audio as both an mp3 and a wav file, as well as a transcript (in csv format) of what is said. Captions are by default white, and set at the top of the screen. 

## Installation
Use the command pip to install the required librairies.
```bash
pip install -r requirements.txt
```


requirements.txt is attached to this project

## Usage
How to add examples? Links maybe to a short sample video with the outputed transcript and another link to the captioned video?

```python
from pydub import AudioSegment
from pathlib import Path

import speech_recognition as sr
import os

import cv2

import pandas as pd
```

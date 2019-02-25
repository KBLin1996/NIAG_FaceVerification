import cv2
import base64
import numpy as np
from Face import Face

a = Face()

with open("kb.png", "rb") as frame:
    base64_image = base64.b64encode(frame.read())

result = a.from_backend('KB', base64_image)

cv2.imwrite("result.jpg", result)

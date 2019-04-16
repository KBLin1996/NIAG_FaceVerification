import io
import os
import cv2
import sys
import glob
import json
import time
import click
import base64
import numpy as np
import face_recognition
from PIL import ImageDraw, ImageFont, Image

class Face(object):

    def from_backend(self, ID, image_src, Name):

        Start = time.time() # Timer Started

        try:
            decode_image = base64.b64decode(image_src)
            img_array = np.fromstring(decode_image, np.uint8)
            self._image = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)

            self._ID = ID
            self._name = Name
            print("Transmission Success !\n")

            self._my_json = {}
            self._my_json["PID"] = self._ID
            self._my_json["Name"] = self._name
            self._my_json["Distance"] = -1
            self._my_json["Prediction"] = "Error"
            final_json = self.face_analyze()
            End = time.time() # Timer Ended
            click.echo("Overall Time: %f sec" % (End - Start))

            return final_json

        except Exception as e:
            print("Transmission Failed\n", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def face_distance(self, known_encode, ID_encode):

        try:
            if len(known_encode) == 0 or len(ID_encode) == 0:
                return np.empty((0))

            return np.linalg.norm([known_encode] - ID_encode, axis=1)[0]

        except Exception as e:
            print("face_Distance Failed\n", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def face_analyze(self):
        try:
            # Load a sample picture and learn how o recognize it.
            face_path = os.getcwd()
            face_path = os.path.join(face_path, "NIAG_Image")
            face_list = glob.glob(os.path.join(face_path, "*"))
            flag = 0

            for face in face_list:

                file_base = os.path.basename(face)

                # To ensure that there exist such a face with self._ID at the dataset
                if os.path.splitext(file_base)[0] == self._ID:

                    flag = 1
                    # face_recognition.load_image_file(file, mode='RGB') -> Loads an image file (.jpg, .png, etc) into a numpy array
                    file_image = face_recognition.load_image_file(face)
                    # Create arrays of known face encodings and their names
                    known_encoding = face_recognition.face_encodings(file_image)
                    ID_encoding = face_recognition.face_encodings(self._image)

                    if len(known_encoding) == 0:
                        click.echo("ERROR: No face found in the dataset image. Print the origin image. Image Name: %s" %file_base)
                        error_msg = "Can not check in with this image"
                        file_path = os.path.join(face_path, file_base)
                        img = cv2.imread(file_path)
                        cv2.putText(img, error_msg, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 0, 255), 2)
                        cv2.imwrite("Result/" + self._ID + ".jpg", img)
                        return self._my_json

                    elif len(ID_encoding) == 0:
                        click.echo("ERROR: No face found on user image. Print the origin image. Image Name: %s" %file_base)
                        error_msg = "No face found on user image"
                        cv2.putText(self._image, error_msg, (60, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        cv2.imwrite("Result/" + self._ID + ".jpg", self._image)
                        return self._my_json

                    if len(known_encoding) > 1:
                        click.echo("ERROR: More than one face found in the dataset image. Considering the first detecting image. Image Name: $s" %file_base)

                    if len(ID_encoding) > 1:
                        click.echo("ERROR: More than one face found on user image. Print the origin image. Image Name: %s" %file_base)

                    known_encoding = known_encoding[0]
                    ID_encoding = ID_encoding[0]

                    # See whether ID is match with the face(s) that we already known
                    distance = self.face_distance(known_encoding, ID_encoding)

            if flag != 1:
                print("No such ID in database !\n")
                distance = 99999

            self._my_json = self.face_drawing(distance)

            return self._my_json

        except Exception as e:
            print("face_Analyze Failed\n", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def face_drawing(self, distance):
        try:
            # Initialize some variables
            name = self._name

            # Returns an array of bounding boxes of human faces in a image
            ID_locations = face_recognition.face_locations(self._image)

            # Display the results
            # zip: https://github.com/KBLin1996/Python_Practice/edit/master/basic_python.py
            for (top, right, bottom, left) in ID_locations:

                # Draw a box around the face, color -> (B,G,R)
                # Change showing color here -> if OK (green), else if Maybe (yellow), else Unknown (red)
                if distance <= 0.45:
                    # cv2.rectangle(frame, vertex's coordinate, diagonal's coordinate, line color, line breadth)
                    cv2.rectangle(self._image, (left, top), (right, bottom), (0, 205, 0), 2)

                    # Draw a label with a name below the face, color -> (B,G,R)
                    cv2.rectangle(self._image, (left, bottom), (right, bottom + 35), (0, 205, 0), cv2.FILLED)
                    prediction = "Pass"

                elif distance > 0.45 and distance <= 0.5:
                    cv2.rectangle(self._image, (left, top), (right, bottom), (0, 215, 255), 2)
                    cv2.rectangle(self._image, (left, bottom), (right, bottom + 35), (0, 215, 255), cv2.FILLED)
                    prediction = "Maybe"

                else:
                    cv2.rectangle(self._image, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(self._image, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)
                    prediction = "Fail"

                if distance >= 0.99:
                    name = name + " (Not Exist)"


                font = ImageFont.truetype('NotoSansCJK-Black.ttc', 26)
                image_PIL = Image.fromarray(cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB))
                draw_name = ImageDraw.Draw(image_PIL)
                draw_name.text((int((left + right) / 2 - 36), bottom - 3), name, fill=(255, 255, 255), font=font)
                self._image = cv2.cvtColor(np.asarray(image_PIL), cv2.COLOR_RGB2BGR)
                

                height, width, dimension = self._image.shape
                self._my_json["Image_shape"] = {}
                self._my_json["Image_shape"]["Height"] = height
                self._my_json["Image_shape"]["Width"] = width
                height = height - int(height / 40)
                width = int(width / 40)

                similar = str(round(1/(1 + distance) *100, 2)) + '%'
                info = "Similarity: " + similar
                # cv2.putText(frame, test, coordinate (text's bottom-left), font, size, text color, text breadth, line options (optional))
                cv2.putText(self._image, info, (width, height), cv2.FONT_HERSHEY_COMPLEX, 1, (40, 200, 255), 2)

                height = height - int(height / 10)
                info = "Result: " + prediction
                cv2.putText(self._image, info, (width, height), cv2.FONT_HERSHEY_COMPLEX, 1, (40, 200, 255), 2)

                ID = self._ID[:3] + '****' + self._ID[7:]
                height = height - int(height / 10)
                info = "ID: " + ID
                cv2.putText(self._image, info, (width, height), cv2.FONT_HERSHEY_COMPLEX, 1, (40, 200, 255), 2)

                self._my_json["Name"] = name
                self._my_json["Distance"] = distance
                self._my_json["Prediction"] = prediction
                self._my_json["Similarity"] = similar

            cv2.imwrite("Result/" + self._ID + ".jpg", self._image)

            return self._my_json

        except Exception as e:
            print("face_Drawing Failed\n", e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


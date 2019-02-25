import io
import os
import cv2
import glob
import time
import click
import base64
import numpy as np
import face_recognition

class Face(object):

    def from_backend(self, ID, image_src):

        self._Start = time.time() # Timer Started

        try:
            decode_image = base64.b64decode(image_src)
            img_array = np.fromstring(decode_image, np.uint8)
            self._image = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
            self._ID = ID
            print("Transmission Success !\n")

            frame = self.face_analyze()

            End = time.time() # Timer Ended
            click.echo("Overall Time: %f sec" % (End - self._Start))

            return frame

        except Exception as e:
            print("Transmission Failed\n", e)


    def face_distance(self, known_encode, ID_encode):

        try:
            if len(known_encode) == 0 or len(ID_encode) == 0:
                return np.empty((0))

            return np.linalg.norm([known_encode] - ID_encode, axis=1)[0]

        except Exception as e:
            print("face_Distance Failed\n", e)


    def face_analyze(self):
        try:
            # Load a sample picture and learn how to recognize it.
            # face_recognition.load_image_file(file, mode='RGB') -> Loads an image file (.jpg, .png, etc) into a numpy array

            face_path = '/home/kb/Keras/Side_Project/Face/Image/'

            face_list = glob.glob(os.path.join(face_path, "*"))

            for face in face_list:

                file_base= os.path.basename(face)

                if file_base[:-4] == self._ID:

                    file_image = face_recognition.load_image_file(face)

	    		    # Create arrays of known face encodings and their names
                    known_encoding = face_recognition.face_encodings(file_image)[0]
                    ID_encoding = face_recognition.face_encodings(self._image)[0]

                    # See whether ID is match with the face(s) that we already known
                    match = face_recognition.compare_faces([known_encoding], ID_encoding, tolerance=0.45)
                    distance = self.face_distance(known_encoding, ID_encoding)

                    frame = self.face_drawing(match, distance)

            return frame

        except Exception as e:
            print("face_Analyze Failed\n", e)


    def face_drawing(self, result, distance):
        try:
            # Initialize some variables
            face_names = []
            name = "Unknown"

            # If matches the specfic face in the dataset
            if True in result:
                # Append the name if matches
                name = self._ID

            face_names.append(name)

            # Returns an array of bounding boxes of human faces in a image
            ID_locations = face_recognition.face_locations(self._image, model="cnn")

            # Display the results
            # zip: https://github.com/KBLin1996/Python_Practice/edit/master/basic_python.py 
            for(top, right, bottom, left), name in zip(ID_locations, face_names):

                # Draw a box around the face, color -> (B,G,R)
                # Change showing color here -> if OK (green), else if Maybe (yellow), else Unknown (red)
                if distance <= 0.45:
                    # cv2.rectangle(frame, vertex's coordinate, diagonal's coordinate, line color, line breadth)
                    cv2.rectangle(self._image, (left, top), (right, bottom), (0, 205, 0), 2)

                    # Draw a label with a name below the face, color -> (B,G,R)
                    cv2.rectangle(self._image, (left, bottom), (right, bottom + 35), (0, 205, 0), cv2.FILLED)

                elif distance > 0.45 and distance <= 0.5:
                    cv2.rectangle(self._image, (left, top), (right, bottom), (0, 215, 255), 2)
                    cv2.rectangle(self._image, (left, bottom), (right, bottom + 35), (0, 215, 255), cv2.FILLED)

                else:
                    cv2.rectangle(self._image, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(self._image, (left, bottom), (right, bottom + 35), (0, 0, 255), cv2.FILLED)

                # cv2.putText(frame, test, coordinate (text's bottom-left), font, size, text color, text breadth, line options (optional))
                cv2.putText(self._image, name, (left + 6, bottom + 29), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

                height, width, dimension = self._image.shape
                height = height - int(height / 40)
                width = int(width / 40)

                info = 'ID:' + self._ID + ', Similarity:'+ str(round(100 - distance * 100, 2)) + '%'
                cv2.putText(self._image, info, (width, height), cv2.FONT_HERSHEY_COMPLEX, 1, (40, 200, 255), 2)

            return self._image

        except Exception as e:
            print("face_Drawing Failed\n", e)

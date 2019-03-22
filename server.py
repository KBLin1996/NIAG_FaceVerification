#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import cv2
import os
import numpy as np
import face_recognition
import time
import flask
import eventlet
from flask import request, jsonify, Flask, render_template
from flask_socketio import SocketIO, Namespace, emit
from Face import Face
from OpenSSL import SSL

app = flask.Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
current_card_id = None
name = None
data = None


@app.route('/', methods=['get'])
def index():
    return app.send_static_file('index.html')

@socketio.on('connect', namespace='/public')
def call_front_end():
    socketio.emit('server_response', {'data':123}, namespace = '/public')


@app.route('/company', methods=['POST'])
def call_by_company():
    global current_card_id
    global name
    global data
    data = request.get_json()
    current_card_id = data['PID']
    name = data['NAME']
    print(current_card_id)
    print(name)
    socketio.emit('server_got_card', {'data':current_card_id}, namespace = '/public')
    return 'Transmission Success!!'


@app.route('/company/result', methods=['POST'])
def result_for_company():
    global data
    return jsonify(data)


@app.route('/recognize', methods=['POST'])
def recognize_face():
    global current_card_id
    global name
    global data
    start = time.time()
    encoded_img = request.form.get('encodedImg')
    print(name)
    a = Face()
    result = a.from_backend(current_card_id, encoded_img[22:], name)
    data = result
    with open("./Result/{}.jpg".format(current_card_id), "rb") as image_file:
        image = b'data:image/jpeg;base64,' + base64.b64encode(image_file.read())
    end = time.time()
    print(end-start)

    return image


@app.route('/current/card/', methods=['POST'])
def set_card_id():
    global current_card_id
    current_card_id = request.form.get('id')

    return 'OK'


def main():
    socketio.run(app, port=5000, debug=True)

if __name__ == '__main__':
    main()

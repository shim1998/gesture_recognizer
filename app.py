from load import *
from MEI_MHI import *
import cv2
from flask import Flask, render_template, Response, jsonify, request
from flask_bootstrap import Bootstrap
from camera import VideoCamera

app = Flask(__name__)
Bootstrap(app)
video_camera = None
global_frame = None

global model, graph
model, graph = init()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera
    if video_camera == None:
        video_camera = VideoCamera()

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        return jsonify(result="stopped")

@app.route('/redirect',methods=["GET","POST"])
def get_gesture():
    createMEIsandMHIs('videos/test.avi')
    return render_template('redirect.html')

@app.route('/gesture',methods=["GET","POST"])
def gesture_recognise():
    image=cv2.imread('mei/test.jpg')
    with graph.as_default():
        ans=model.predict(image.reshape(-1,90,67,1))
    if(np.argmax(ans) == 0):
        output="Arm Waving Sideways"
    elif (np.argmax(ans) == 1):
        output="Right Hand Waving"
    elif (np.argmax(ans) == 2):
        output="Arm waving forwards"
    return render_template('predict.html',output=output)

def video_stream():
    global video_camera
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera(0)

    while True:
        frame = video_camera.get_frame()

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')

@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)

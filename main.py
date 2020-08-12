import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pathlib
import matplotlib.pyplot as plt

import os
import django
from django.utils import timezone

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'EmoDet.settings')
django.setup()

from main.models import Person, PlotStats


def crop_center(img, x, y, w, h):
    return img[y:y + h, x:x + w]


def preprocess_img(raw):
    img = cv2.resize(raw, (200, 200))
    img = np.expand_dims(img, axis=0)
    if np.max(img) > 1:
        img = img / 255.0
    return img


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "key doesn't exist"


def predict_emotion(model, inputs, outputs, raw, x, y, w, h):
    img = crop_center(raw, x, y, w, h)
    img = preprocess_img(img)
    model.set_tensor(inputs['index'], img.astype(np.float32))
    model.invoke()
    res = model.get_tensor(outputs['index'])
    classes = np.argmax(res, axis=1)
    if classes == 0:
        emotion = 'anger'
    elif classes == 1:
        emotion = 'disgust'
    elif classes == 2:
        emotion = 'fear'
    elif classes == 3:
        emotion = "happy"
    elif classes == 4:
        emotion = "neutral"
    elif classes == 5:
        emotion = 'sadness'
    else:
        emotion = 'surprised'
    return emotion, np.max(res)


def main():
    home_dir = pathlib.Path.home()
    cap = cv2.VideoCapture("wws.mp4")
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # ordering the emotions from worst to best in a dictionary

    emotion_lut = {
        "anger": 1,
        "fear": 2,
        "disgust": 3,
        "sadness": 4,
        "neutral": 5,
        "happy": 6,
        "surprised": 7
    }

    avg_overall = []
    avg_emotions = []

    model = tflite.Interpreter("tfmodels/model_optimized.tflite")
    model.allocate_tensors()
    inputs = model.get_input_details()[0]
    outputs = model.get_output_details()[0]

    # Using HaarCascade classifier
    face_detect = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    while True:
        frame_emotions = None
        face_id = 0
        success_reading, image = cap.read()

        '''
        scale_percent = 50  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        '''

        if success_reading:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # detect faces in our gray picture
            faces = face_detect.detectMultiScale(img_gray,
                                                 scaleFactor=1.3,
                                                 minNeighbors=5
                                                 )
            for (x, y, w, h) in faces:
                frame_emotions = np.zeros((len(faces), 2))
                face_id += 1
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                prediction, probability = predict_emotion(model, inputs, outputs, image, x, y, w, h)
                to_display = prediction + ' ' + str(int(probability * 100))
                frame_emotions[face_id - 1, 0] = emotion_lut[prediction]
                frame_emotions[face_id - 1, 1] = round((probability * 100), 2)

                if probability > 0.90:
                    cv2.putText(image, to_display, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2, cv2.LINE_AA)

                    timestamp = timezone.localtime(timezone.now()).time()

                    cv2.imwrite(
                        str(home_dir) + "/Desktop/PPM Project/EmoDet/main/faces_pics/" +
                        str(face_id) + "_" + str(timestamp) + ".jpg",
                        image[y:y + h, x:x + w])

                    person = Person(person_number=face_id,
                                    person_emotion=prediction,
                                    person_last_seen=timezone.localtime(timezone.now()),
                                    person_prediction_prob=str(probability * 100)[:4]
                                    )

                    person.person_thumbnail = str(face_id) + "_" + str(timestamp) + ".jpg"
                    person.save()

            if frame_emotions is not None:
                average_emotion = round(np.average(frame_emotions[:, 0], weights=frame_emotions[:, 1]))
                average_emotion = get_key(average_emotion, emotion_lut)
                avg_emotions.append(average_emotion)

                plot1 = PlotStats(plot_id=1)
                x_axis = np.arange(0, len(avg_emotions))
                plt.title("Average Emotions over Frames")
                plt.xlabel("Frames")
                plt.ylabel("Average Emotion")

                plt.grid()
                plt.plot(x_axis, avg_emotions)
                plt.savefig(str(home_dir) +
                            "/Desktop/PPM Project/EmoDet/main/faces_pics/graph1.jpg", bbox_inches='tight')
                plt.clf()
                plot1.plot = "graph1.jpg"
                plot1.save()

                plot2 = PlotStats(plot_id=2)
                sum_tmp = 0
                for emotion in avg_emotions:
                    sum_tmp += emotion_lut[emotion]

                average_over_time = round((sum_tmp / len(avg_emotions)))
                avg_overall.append(get_key(average_over_time, emotion_lut))

                plt.title("Average Emotions over Time")
                plt.xlabel("Frames")
                plt.ylabel("Average Emotion")

                plt.grid()
                plt.plot(x_axis, avg_overall)
                plt.savefig(str(home_dir) +
                            "/Desktop/PPM Project/EmoDet/main/faces_pics/graph2.jpg", bbox_inches='tight')

                plot2.plot = "graph2.jpg"
                plot2.save()
                plt.clf()

            cv2.imshow("Faces & Emotions", image)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    cap.release()


if __name__ == "__main__":
    main()

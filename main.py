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

from main.models import Person, PlotStats, AIPrediction


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


def predict_emotion(model, inputs, outputs, raw, x, y, w, h, classes_lut):
    img = crop_center(raw, x, y, w, h)
    img = preprocess_img(img)
    model.set_tensor(inputs['index'], img.astype(np.float32))
    model.invoke()
    res = model.get_tensor(outputs['index'])
    emotion = classes_lut[int(np.argmax(res, axis=1))]
    all_model_results = {classes_lut[i]: round(res[0, i], 2) * 100 for i in range(7)}

    return emotion, np.max(res), all_model_results


def plot_emotion_probabilities(emotion_lut, scene_mood):
    """
        we want to represent the mean probability for each emotion in a
        graph that gets updated after each frame that contains an emotion
    """
    plt.title("Mood of the scene")
    plt.xlabel("Frames")
    plt.ylabel("Average of each emotion")

    plot1 = PlotStats(plot_id=1)
    plt.grid()

    max_dim = 0
    for i in range(7):
        if len(scene_mood[i]) > max_dim:
            max_dim = len(scene_mood[i])

    x_axis = np.arange(max_dim)

    for i in range(7):
        plt.plot(x_axis, scene_mood[i], label=str(emotion_lut[i]))

    plt.legend()

    plt.savefig(str(pathlib.Path.home()) +
                "/Desktop/PPM Project/EmoDet/main/pics/graph1.jpg", bbox_inches='tight')
    plt.clf()
    plot1.plot = "graph1.jpg"
    plot1.save()


def plot_histogram(face_id, all_model_results, ai_prediction_django_model):
    plt.bar(list(all_model_results.keys()), all_model_results.values(), color='g')
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Emotions')
    plt.ylabel('Probability')
    plt.title('NeuralNetwork Predictions')
    maxprob = np.max(list(all_model_results.values()))
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxprob / 10) * 10 if maxprob % 10 else maxprob + 10)

    plt.savefig(str(pathlib.Path.home()) +
                "/Desktop/PPM Project/EmoDet/main/pics/hist"
                + str(face_id) + ".jpg", bbox_inches='tight')
    plt.clf()

    ai_prediction_django_model.plot = "hist" + str(face_id) + ".jpg"


def main():
    home_dir = pathlib.Path.home()
    cap = cv2.VideoCapture("wws.mp4")
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # ordering the emotions from worst to best in a dictionary

    emotion_lut = {0: 'anger',
                   1: 'disgust',
                   2: 'fear',
                   3: 'happy',
                   4: 'neutral',
                   5: 'sadness',
                   6: 'surprised'}

    model = tflite.Interpreter("tfmodels/model_optimized.tflite")
    model.allocate_tensors()
    inputs = model.get_input_details()[0]
    outputs = model.get_output_details()[0]

    # Using HaarCascade classifier
    face_detect = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    scene_mood = [[] for i in range(7)]
    face_id = 0  # is the primary key in Person Table of our DataBase
    num_frames = 0  # counts the frames

    while True:
        face_number_in_frame = 0  # counts the number of faces in each frame
        model_results = [[] for i in range(7)]
        face_detected = False
        success_reading, image = cap.read()
        num_frames += 1

        if success_reading:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # detect faces in our gray picture
            faces = face_detect.detectMultiScale(img_gray,
                                                 scaleFactor=1.3,
                                                 minNeighbors=5
                                                 )
            for (x, y, w, h) in faces:
                face_number_in_frame += 1
                face_id += 1
                face_detected = True
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                top_prediction, top_probability, all_model_results = predict_emotion(model,
                                                                                     inputs,
                                                                                     outputs,
                                                                                     image,
                                                                                     x, y, w, h,
                                                                                     emotion_lut)
                to_display = top_prediction + ' ' + str(int(top_probability * 100))

                cv2.putText(image, to_display, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)

                timestamp = timezone.localtime(timezone.now()).time()

                cv2.imwrite(
                    str(home_dir) + "/Desktop/PPM Project/EmoDet/main/pics/" +
                    str(face_id) + "_" + str(top_prediction) + "_" + str(timestamp) + ".jpg",
                    image[y:y + h, x:x + w])

                person = Person(
                    person_id=face_id,
                    person_number_in_last_frame=face_number_in_frame,
                    person_frame=num_frames,
                    person_emotion=top_prediction,
                    person_last_seen=timezone.localtime(timezone.now()),
                    person_prediction_prob=str(top_probability * 100)[:4]
                )

                person.person_thumbnail = str(face_id) + "_" + str(top_prediction) + "_" + str(timestamp) + ".jpg"
                person.save()

                # model_results contains all the results from the model in this frame
                # this allows to do averages if multiple faces have more than one prediction higher than 0
                for emotion, probability in zip(np.arange(7), list(all_model_results.values())):
                    model_results[emotion].append(probability)


                # adding the predictions to database
                ai_prediction = AIPrediction(
                    person=person,
                    anger=round(all_model_results["anger"], 4),
                    disgust=round(all_model_results["disgust"], 4),
                    fear=round(all_model_results["fear"], 4),
                    happy=round(all_model_results["happy"], 4),
                    neutral=round(all_model_results["neutral"], 4),
                    sadness=round(all_model_results["sadness"], 4),
                    surprised=round(all_model_results["surprised"], 4)
                )
                ai_prediction.plot = "hist" + str(face_id) + ".jpg"
                ai_prediction.save()

                plot_histogram(face_id, all_model_results, ai_prediction)

            # print(model_results)

            if face_detected:
                for i in range(len(scene_mood)):
                    scene_mood[i].append(np.average(model_results[i]))
                plot_emotion_probabilities(emotion_lut, scene_mood)

            # print(scene_mood)

            cv2.imshow("Faces & Emotions", image)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            break

    cap.release()


if __name__ == "__main__":
    main()

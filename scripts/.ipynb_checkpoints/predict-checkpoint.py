import pickle
import uuid
import datetime
import numpy as np

model_file_name = "/mnt/code/models/sklearn_gbm.pkl"
model = pickle.load(open(model_file_name, 'rb'))

# from domino_prediction_logging.prediction_client import PredictionClient
from domino_data_capture.data_capture_client import DataCaptureClient

features = ['density', 'volatile_acidity', 'chlorides', 'is_red', 'alcohol']

target = ["quality"]

# pred_client = PredictionClient(features, target)
data_capture_client = DataCaptureClient(features, target)

def predict(density, volatile_acidity, chlorides, is_red, alcohol, wine_id=None):
    feature_values = [density, volatile_acidity, chlorides, is_red, alcohol]
    prediction = model.predict([feature_values]).tolist()


    # Record eventID and current time
    if wine_id is None:
        print("No ID found! Creating a new one.")
        wine_id = str(datetime.datetime.now())
        # custid = uuid.uuid4()
    print('Wine ID is: {}'.format(wine_id))

    # pred_client.record(feature_values, prediction, event_id=custid)
    data_capture_client.capturePrediction(feature_values, prediction,event_id=wine_id)

    return dict(prediction=prediction[0])
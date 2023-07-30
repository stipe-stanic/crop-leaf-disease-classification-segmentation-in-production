import requests
import deployment.local.config as local_config

from PIL import Image


def response_from_server(url: str, image_file: Image, verbose: bool = True) -> requests.models.Response:
    """Makes a POST request to the server and returns the response.

    :param url: URL that the request is sent to.
    :param image_file: File to upload, should be an image.
    :param verbose: True if the status of the response should be printed.
    :returns requests.models.Response: Response from the server.
    """

    files = {'file': image_file}
    response = requests.post(url, files=files)
    status_code = response.status_code

    if verbose:
        msg = "Success" if status_code == 200 else "There was an error when handling the request."
        print(msg)
    return response


if __name__ == '__main__':
    base_url = local_config.base_url
    endpoint = '/predict'
    model = 'res-model'

    full_url = base_url + endpoint + '?model=' + model

    with open('images_post/corn/gray_spot_corn_4.jpg', 'rb') as image_file:
        prediction = response_from_server(full_url, image_file)

    response_json = prediction.json()
    prediction_label = response_json['prediction']
    confidence = response_json['confidence']

    print(f'Class: {prediction_label}\nConfidence: {confidence}')

import requests
import deployment.local.config as local_config

def response_from_server(url, image_file, verbose=True) ->  requests.models.Response:
    """Makes a POST request to the server and returns the response.

    Args:
        url (str): URL that the request is sent to.
        image_file (_io.BufferedReader): File to upload, should be an image.
        verbose (bool): True if the status of the response should be printed. False otherwise.

    Returns:
        requests.models.Response: Response from the server.
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

    with open('images_post/apple_cedar_apple_rust5.jpg', 'rb') as image_file:
        prediction = response_from_server(full_url, image_file)

    response_json = prediction.json()
    prediction_label = response_json['prediction']
    print(prediction_label)


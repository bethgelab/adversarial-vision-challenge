import sys
from abc import abstractmethod

import requests
import numpy as np
from foolbox.models import Model
import bson
import os

from .retry_helper import retryable
from .logger import logger
from .common import check_image, _assert

if sys.version_info > (3, 3):
    import urllib.parse as parse
else:
    import urlparse as parse


class HTTPClient(object):
    """Base class for HTTPModel and HTTPAttack."""

    def _encode_array_data(self, array):
        """
        Converts a numpy array to bytes.
        """
        return array.tobytes()

    def _decode_array_data(self, data, dtype, shape):
        """
        Converts bytes to a numpy array.
        """
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    def _encode_arrays(self, data):
        """
        Can be used by subclasses to encode numpy arrays. To use it,
        subclasses must implement _encode_array_data.
        """
        encoded = {}
        for key in list(data.keys()):
            if isinstance(data[key], np.ndarray):
                array = data[key]
                encoded[key] = {
                    'type': 'array',
                    'shape': array.shape,
                    'dtype': array.dtype.str,
                    'data': self._encode_array_data(array),
                }
            else:
                encoded[key] = data[key]
        return encoded

    def _decode_arrays(self, encoded):
        """
        Can be used by subclasses to decode numpy arrays. To use it,
        subclasses must implement _decode_array_data.
        """
        decoded = {}
        for key in list(encoded.keys()):
            if hasattr(encoded[key], 'get') \
                    and encoded[key].get('type') == 'array':
                shape = encoded[key]['shape']
                dtype = encoded[key]['dtype']
                data = encoded[key]['data']
                array = self._decode_array_data(data, dtype, shape)
                decoded[key] = array
            else:
                decoded[key] = encoded[key]
        return decoded

    @retryable
    def _post(self, path, data):
        """
        Encodes the data dictionary, sends it to the http server using
        an http post request to the url specified by path, decodes
        the result and returns it as a dictionary.
        """
        url = self._url(path=path)
        
        headers = {'content-type': 'application/bson'}
        eval_secret = os.getenv('EVALUATOR_SECRET')
        if eval_secret is not None:
            headers['Evaluator-Secret'] = eval_secret

        data = self._encode_arrays(data)
        data = bson.dumps(data)
        r = self.requests.post(url, headers=headers, data=data)
        r.raise_for_status()
        assert r.ok
        result = r.content
        result = bson.loads(result)
        result = self._decode_arrays(result)
        return result

    @retryable
    def _get(self, path):
        """
        Performs a get request to the url specified by path and
        returns the result as text. Subclasses can override this
        if necessary.
        """
        url = self._url(path=path)
        r = self.requests.get(url)
        r.raise_for_status()
        assert r.ok
        return r.text

    @abstractmethod
    def _url(self, path=''):
        raise NotImplementedError


class TinyImageNetBSONModel(Model, HTTPClient):
    """Base class for models that connect to an http server and
    dispatch all requets to that server.

    Parameters
    ----------
    url : str
        The http or https URL of the server.

    """

    def __init__(self, url):
        self.requests = requests

        self._base_url = url

        super(TinyImageNetBSONModel, self).__init__(
            bounds=(0, 255), channel_axis=3)

    def _url(self, path=''):
        return parse.urljoin(self._base_url, path)

    @property
    def base_url(self):
        return self._base_url

    def server_version(self):
        return self._get('/server_version')

    def __call__(self, image):
        return self.predict(image)

    def predict(self, image):
        image = check_image(image)
        data = {'image': image}
        result = self._post('/predict', data)

        prediction = result['prediction']
        _assert(isinstance(prediction, int), "prediction should return an int value, but got: %s" % type(prediction))
        _assert((0 <= prediction < 200), "prediction should be a value between 0 and 200, but got: %s" % prediction)
        return prediction

    def batch_predictions(self, images):
        assert images.shape[0] == 1
        image = images[0]
        predictions = self.predictions(image)
        predictions = predictions[np.newaxis]
        return predictions

    def predictions(self, image):
        class_ = self.predict(image)
        predictions = np.zeros((200,), dtype=np.float32)
        predictions[class_] = 1
        return predictions

    def num_classes(self):
        return 200

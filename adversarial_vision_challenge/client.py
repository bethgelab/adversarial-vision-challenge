from abc import abstractmethod
import sys

if sys.version_info > (3, 3):
    import urllib.parse as parse
else:
    import urlparse as parse

import numpy as np

from foolbox.models import DifferentiableModel
from foolbox.attacks import Attack
import foolbox


class HTTPClient(object):
    """Base class for HTTPModel and HTTPAttack."""

    @abstractmethod
    def _encode_array_data(self, array):
        """
        Must be implemented by subclasses that use _encoded_arrays.
        """
        raise NotImplementedError

    @abstractmethod
    def _decode_array_data(self, data, dtype, shape):
        """
        Must be implemented by subclasses that use _decode_arrays.
        """
        raise NotImplementedError

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

    @abstractmethod
    def _post(self, path, data):
        """
        Encodes the data dictionary, sends it to the http server using
        an http post request to the url specified by path, decodes
        the result and returns it as a dictionary.
        """
        raise NotImplementedError

    def _get(self, path):
        """
        Performs a get request to the url specified by path and
        returns the result as text. Subclasses can override this
        if necessary.
        """
        url = self._url(path=path)
        r = self.requests.get(url)
        assert r.ok
        return r.text

    @abstractmethod
    def _url(self, path=''):
        raise NotImplementedError


class HTTPModel(DifferentiableModel, HTTPClient):
    """Base class for models that connect to an http server and
    dispatch all requets to that server.

    Parameters
    ----------
    url : str
        The http or https URL of the server.

    """

    def __init__(self, url):
        import requests
        self.requests = requests

        self._base_url = url

        bounds = self._remote_bounds()
        channel_axis = self._remote_channel_axis()

        super(HTTPModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis)

        self._dataset = self._remote_dataset()
        self._image_size = self._remote_image_size()
        self._channel_order = self._remote_channel_order()
        self._num_classes = self._remote_num_classes()

    def _url(self, path=''):
        return parse.urljoin(self._base_url, path)

    @property
    def base_url(self):
        return self._base_url

    def _remote_bounds(self):
        s = self._get('/bounds')
        min_, max_ = s.split('\n')
        min_ = float(min_)
        max_ = float(max_)
        return (min_, max_)

    def _remote_channel_axis(self):
        s = self._get('/channel_axis')
        return int(s)

    def _remote_image_size(self):
        s = self._get('/image_size')
        return int(s)

    def _remote_dataset(self):
        return self._get('/dataset').upper()

    def _remote_channel_order(self):
        return self._get('/channel_order')

    def _remote_num_classes(self):
        s = self._get('/num_classes')
        return int(s)

    def shutdown(self):
        s = self._get('/shutdown')
        return s

    def num_classes(self):
        return self._num_classes

    def channel_order(self):
        return self._channel_order

    def image_size(self):
        return self._image_size

    def dataset(self):
        return self._dataset

    def server_version(self):
        s = self._get('/server_version')
        return s

    def batch_predictions(self, images):
        images = np.asarray(images)
        data = {'images': images}
        result = self._post('/batch_predictions', data)
        predictions = result['predictions']
        return predictions

    def predictions_and_gradient(self, image, label):
        image = np.asarray(image)
        label = np.asarray(label)
        data = {'image': image, 'label': label}
        result = self._post('/predictions_and_gradient', data)
        predictions = result['predictions']
        gradient = result['gradient']
        return predictions, gradient

    def backward(self, gradient, image):
        gradient = np.asarray(gradient)
        image = np.asarray(image)
        data = {'gradient': gradient, 'image': image}
        result = self._post('/backward', data)
        gradient = result['gradient']
        return gradient


class HTTPAttack(Attack, HTTPClient):
    """Base class for attacks that connect to an http server and
    dispatch all requets to that server.

    Parameters
    ----------
    url : str
        The http or https URL of the server.

    """

    def __init__(self, attack_url, model=None, criterion=None):
        import requests
        self.requests = requests

        self._base_url = attack_url

        super().__init__(model=model, criterion=criterion)

    def _url(self, path=''):
        return parse.urljoin(self._base_url, path)

    def shutdown(self):
        s = self._get('/shutdown')
        return s

    def server_version(self):
        s = self._get('/server_version')
        return s

    def _apply(self, a):
        assert a.image is None
        assert a.distance.value == np.inf
        assert a._distance == foolbox.distances.MSE
        assert isinstance(a._criterion, foolbox.criteria.Misclassification)  # noqa: E501
        assert isinstance(a._model, BSONModel)

        image = np.asarray(a.original_image)
        label = np.asarray(a.original_class)
        model_url = a._model.base_url
        criterion_name = 'Misclassification'

        data = {
            'model_url': model_url,
            'image': image,
            'label': label,
            'criterion_name': criterion_name,
        }
        result = self._post('/run', data)
        adversarial_image = result['adversarial_image']

        if adversarial_image is not None:
            a.predictions(adversarial_image)
            assert a.image is not None
            assert a.distance.value < np.inf


class BSON(object):

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

    def _post(self, path, data):
        import bson
        url = self._url(path=path)
        headers = {'content-type': 'application/bson'}
        data = self._encode_arrays(data)
        data = bson.dumps(data)
        r = self.requests.post(url, headers=headers, data=data)
        assert r.ok
        result = r.content
        result = bson.loads(result)
        result = self._decode_arrays(result)
        return result


class BSONAttack(BSON, HTTPAttack):
    """
    An attack that connects to an http server and dispatches all
    requests to that server using BSON-encoded http requests.
    """
    pass


class BSONModel(BSON, HTTPModel):
    """
    A model that connects to an http server and dispatches all
    requests to that server using BSON-encoded http requests.
    """
    pass

import time
import os
from .logger import logger
from .notifier import CrowdAiNotifier
import threading
import uuid


class NoClientInteractionError(Exception):
    pass


class InteractionVerifier:
    __last_request = None
    __time_out = None
    __instance_id = None

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
            cls.instance.start()
        return cls.instance

    def __init__(self):
        # timeout in seconds or client <-> server interaction
        self.__time_out = int(os.getenv('CS_INTERACTION_TIMEOUT', 180))
        self.__instance_id = uuid.uuid1()
        logger.info('client <-> server interaction verifier id: %s', self.__instance_id)


    def start(self):
        self.__last_request = time.time()
        Caller()
        logger.info('Client <-> Server interaction monitor started...')

    def mark(self):
        self.__last_request = time.time()

    def verify(self):
        now = time.time()
        duration = now - self.__last_request
        if duration > self.__time_out:
            logger.error('Client has not sent any requests to to the server for more than %ss', duration)
            CrowdAiNotifier.no_client_interaction()
            raise NoClientInteractionError(
                "Client has not sent any requests to to the server for more than {}s ".format(duration)
            )


class Caller:

    def __init__(self):
        # interval in seconds to check client <-> server interaction
        self.__cs_interaction_verifier = InteractionVerifier()
        self.__interval = int(os.getenv('CS_INTERACTION_CHECK_INTERVAL', 5))
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True
        thread.start()

    def run(self):
        while True:
            self.__cs_interaction_verifier.verify()
            time.sleep(self.__interval)

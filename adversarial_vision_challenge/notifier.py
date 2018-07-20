import crowdai_api

from enum import Enum


class ModelNotifications:
    TYPE = "AVC.MODEL"
    TOO_MANY_REQUESTS = "AVC.MODEL.TOO_MANY_REQUESTS_ERROR"
    NO_CLIENT_INTERACTION = "AVC.MODEL.NO_CLIENT_INTERACTION"


class AttackNotifications:
    TYPE = "AVC.ATTACK"
    COMPLETE = "AVC.ATTACK.COMPLETE"
    RETRIES_EXCEEDED = "AVC.ATTACK.RETRIES_EXCEEDED_ERROR"
    STORE_ADVERSARIAL = "AVC.ATTACK.STORE_ADVERSARIAL"


class GeneralNotifications:
    TYPE = "AVC.GENERAL"
    ASSESTION_FAILURE = "AVC.ASSESTION_FAILURE"


class CrowdAiNotifier():

    @staticmethod
    def _send_notification(event_type, message, payload={}, blocking=False):
        crowdai_events = crowdai_api.events.CrowdAIEvents()
        default_payload = {"challenge_id": "NIPS18_AVC"}
        default_payload.update(payload)
        crowdai_events.register_event(event_type, message, payload, blocking)

    # ~~~~~~~~~~~~~~~~ ATTACK NOTIFICATIONS ~~~~~~~~~~~~~~~~
    @staticmethod
    def store_adversarial(filename):
        CrowdAiNotifier._send_notification(
            event_type=AttackNotifications.STORE_ADVERSARIAL,
            message="",
            payload={
                "type": AttackNotifications.TYPE,
                "filename" : filename
            }
        )

    @staticmethod
    def attack_complete():
        CrowdAiNotifier._send_notification(
            event_type=AttackNotifications.COMPLETE,
            message="Attack successfully completed.",
            payload={
                "type": AttackNotifications.TYPE
            },
            blocking=True
        )

    @staticmethod
    def retries_exceeded():
        CrowdAiNotifier._send_notification(
            event_type=AttackNotifications.RETRIES_EXCEEDED,
            message="No proper response from model after retrying multiple times.",
            payload={
                "type": AttackNotifications.TYPE
            }
        )

    # ~~~~~~~~~~~~~~~~ MODEL NOTIFICATIONS ~~~~~~~~~~~~~~~~

    @staticmethod
    def too_many_requests():
        CrowdAiNotifier._send_notification(
            event_type=ModelNotifications.TOO_MANY_REQUESTS,
            message="The attack has exceeded the max number of allowed predictions requests.",
            payload={
                "type": ModelNotifications.TYPE
            }
        )

    @staticmethod
    def no_client_interaction():
        CrowdAiNotifier._send_notification(
            event_type=ModelNotifications.NO_CLIENT_INTERACTION,
            message="The model server has not received any requests from the client for too long.",
            payload={
                "type": ModelNotifications.TYPE
            }
        )

    @staticmethod
    def assertion_failure(message):
        CrowdAiNotifier._send_notification(
            event_type=GeneralNotifications.ASSESTION_FAILURE,
            message=message,
            payload={
                "type": GeneralNotifications.TYPE
            }
        )

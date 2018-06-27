import crowdai_api

from enum import Enum


class ModelNotifications:
    TYPE = "MODEL"
    TOO_MANY_REQUESTS = "TOO_MANY_REQUESTS_ERROR"


class AttackNotifications:
    TYPE = "ATTACK"
    COMPLETE = "ATTACK_COMPLETE"
    RETRIES_EXCEEDED = "RETRIES_EXCEEDED_ERROR"


class CrowdAiNotifier():

    @staticmethod
    def _send_notification(event_type, message, payload={}):
        crowdai_events = crowdai_api.events.CrowdAIEvents()
        default_payload = {"challenge_id": "NIPS18_AVC"}
        final_payload = default_payload.update(payload)
        crowdai_events.register_event(event_type, message, final_payload)

    # ~~~~~~~~~~~~~~~~ ATTACK NOTIFICATIONS ~~~~~~~~~~~~~~~~

    @staticmethod
    def attack_complete():
        CrowdAiNotifier._send_notification(
            event_type=AttackNotifications.COMPLETE,
            message="Attack successfully completed.",
            payload={
                "type": AttackNotifications.TYPE
            }
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

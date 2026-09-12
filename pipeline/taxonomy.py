"""
Intent taxonomy for the Uber_Support agent.

PROVISIONAL — derived from a first look at the data (screenshots + stats.json),
not from the full 100-thread read. Revise after reading data/uber_threads.jsonl.
Every intent has a one-line definition and two real-style examples used verbatim
in the enrich prompt. Keep intents mutually exclusive and observable from text.
"""
from enum import Enum


class Intent(str, Enum):
    FARE_DISPUTE = "fare_dispute"
    TRIP_OR_DRIVER_ISSUE = "trip_or_driver_issue"
    APP_OR_ACCOUNT_ISSUE = "app_or_account_issue"
    DRIVER_ONBOARDING_OR_EARNINGS = "driver_onboarding_or_earnings"
    UBER_EATS = "uber_eats"
    LOST_ITEM = "lost_item"
    POLICY_OR_INFO_QUESTION = "policy_or_info_question"
    OTHER = "other"


INTENT_DEFINITIONS: dict[Intent, dict] = {
    Intent.FARE_DISPUTE: {
        "definition": "Disputes a charge: wrong fare, double charge, cancellation fee, surge, tip, "
                      "or wants money back for a trip.",
        "examples": [
            "you need to correct false charges from a trip in Lexington KY on Saturday. I want my $13 back",
            "I accidentally started trip but rider never got in the car & I don't know how to get the charge refunded",
        ],
        "ask_for": "the trip date and city so the fare can be reviewed",
    },
    Intent.TRIP_OR_DRIVER_ISSUE: {
        "definition": "Problem with a ride: driver no-show, wrong route, detour, wrong destination, "
                      "driver cancelling on arrival, rude or unsafe driver, vehicle condition, long wait, stranded.",
        "examples": [
            "my driver just drove me to the department of air travel instead of the airport",
            "1 hour+ inside an old Altima that smells and has ripped stain seats, with an inexperienced driver",
        ],
        "ask_for": "the trip date and a short description of what happened",
    },
    Intent.APP_OR_ACCOUNT_ISSUE: {
        "definition": "Cannot use the app or account: login, verification codes, app errors, "
                      "payment method failing, account locked.",
        "examples": [
            "wtf is this.. every time I try to use the app it gives me this crap",
            "I'm not getting your texts",
        ],
        "ask_for": "the email on the account",
    },
    Intent.DRIVER_ONBOARDING_OR_EARNINGS: {
        "definition": "From a driver or applicant: signup status, background check, documents, "
                      "payouts, bank details, missing earnings.",
        "examples": [
            "I've been applying with Uber for 1 1/2 month to be a driver. I haven't any updates yet!",
            "I forgot to change my bank account information and my earnings has already cashed out. What do I do?",
        ],
        "ask_for": "the email used on the driver application",
    },
    Intent.UBER_EATS: {
        "definition": "Food delivery order: missing or wrong items, late or never delivered, "
                      "delivery area, Eats promo not applied.",
        "examples": [
            "Welp... I guess that's one bibimbap order I'm never gonna receive",
            "do you also support ubereats singapore? im reaching out here bec the support there is useless",
        ],
        "ask_for": "the order number or the email on the Eats account",
    },
    Intent.LOST_ITEM: {
        "definition": "Left something in a vehicle; wants it back or to reach the driver.",
        "examples": [
            "left my phone in the back seat of my ride 20 minutes ago, how do I reach the driver",
            "driver has my wallet, I've called twice no answer",
        ],
        "ask_for": "the trip date",
    },
    Intent.POLICY_OR_INFO_QUESTION: {
        "definition": "General question, no incident: pricing, ride pass, city availability, "
                      "driver requirements, promotions.",
        "examples": [
            "how do I get the ride pass, my friend has it and I don't",
            "is Uber available in Coimbatore yet?",
        ],
        "ask_for": "nothing — answer from the help centre",
    },
    Intent.OTHER: {
        "definition": "Rant, compliment, spam, unrelated, or too short to classify.",
        "examples": [
            "Thanks! I appreciate it. Not sure what happened.",
            "Bruh 😭😭😭",
        ],
        "ask_for": "nothing",
    },
}

# Escalation configuration — PROVISIONAL, set from golden-set results later.
AMOUNT_LIMIT: float = 10.0
AUTO_ALLOW: set[Intent] = {
    Intent.POLICY_OR_INFO_QUESTION,
    Intent.DRIVER_ONBOARDING_OR_EARNINGS,
    Intent.APP_OR_ACCOUNT_ISSUE,
    Intent.LOST_ITEM,
    Intent.OTHER,  # canned reply in run.py, no draft: escalating "thanks" wastes a specialist
}
"""Five real-style Uber replies with the deflection/informative label the regex must give them."""
import pytest

from scripts.split_replies import classify_reply

CASES = [
    ("We're here to help! Send us a DM with your email address so we can connect", "deflection"),
    ("Sorry for the trouble! Send us a note here, and our team will be in touch", "deflection"),
    ("Hi there! Please reach out to and fill out the 4 boxes at the bottom", "deflection"),
    ("Ride Pass is currently available in select cities; you'll see it in the app if it's offered near you", "informative"),
    # phrases added after eyeballing the first split
    ("Our team will be following up with you further via email", "deflection"),
    ("We have routed this to the appropriate team who will be in contact shortly", "deflection"),
    ("We'll continue working with you via in app support", "deflection"),
    ("Please respond directly to the in-app messaging thread", "deflection"),
    ("Here to help. Just reach this link: and send a note if a lost item happens", "deflection"),
    ("Here to help! Simply follow the link to update your rating;", "deflection"),
    ("You can check this link; to learn more about Dynamic pricing", "deflection"),
    ("For more info about UberEats availability click here;", "deflection"),
    ("Be sure to check out for more information", "deflection"),
    ("Your trip fare was adjusted because the driver took a longer route; the difference has been refunded", "informative"),
]


@pytest.mark.parametrize("reply,expected", CASES)
def test_classify_reply(reply: str, expected: str) -> None:
    assert classify_reply(reply) == expected


def test_dm_is_word_bounded() -> None:
    assert classify_reply("The admin team has updated your account") == "informative"

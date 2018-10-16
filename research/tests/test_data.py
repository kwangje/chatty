from research.data import daily_dialogue


def test_daily_dialogue_data():
    """Test to display and check that the output format is as expected"""
    data = daily_dialogue.data()
    expected = [
        {'conv_id': 0,
         'utter_id': 0,
         'utterance': 'The kitchen stinks . ',
         'act': 'directive',
         'emotion': 'disgust'},
         {'conv_id': 0,
         'utter_id': 1,
         'utterance': " I'll throw out the garbage . ",
         'act': 'commissive',
         'emotion': 'no emotion'},
         {'conv_id': 1,
         'utter_id': 0,
         'utterance': 'So Dick , how about getting some coffee for tonight ? ',
         'act': 'directive',
         'emotion': 'happiness'},
         {'conv_id': 1,
         'utter_id': 1,
         'utterance': ' Coffee ? I don ’ t honestly like that kind of stuff . ',
         'act': 'commissive',
         'emotion': 'disgust'},
         {'conv_id': 1,
         'utter_id': 2,
         'utterance': ' Come on , you can at least try a little , besides your cigarette . ',
         'act': 'directive',
         'emotion': 'no emotion'},
    ]
    assert data[:5] == expected
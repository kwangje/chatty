import os


_CUR_DIR = os.path.realpath(os.path.dirname(__file__))
_DATA_DIR = os.path.join(_CUR_DIR, 'daily_dialogue')
_EMO_LOOKUP = {
    '0': 'no emotion',
    '1': 'anger',
    '2': 'disgust',
    '3': 'fear',
    '4': 'happiness',
    '5': 'sadness',
    '6': 'surprise'
}
_ACT_LOOKUP = {
    '1': 'inform',
    '2': 'question',
    '3': 'directive',
    '4': 'commissive'
}


__all__ = ['data']


def _load_data(fname):
    with open(os.path.join(_DATA_DIR, fname), 'r') as f:
        return f.readlines()


def _get_dialogues():
    "return dialogues and associated act and emo tags"
    dialogues = _load_data('dialogues_text.txt')
    acts = _load_data('dialogues_act.txt')
    emos = _load_data('dialogues_emotion.txt')
    records = []
    for dial, act, emo in zip(dialogues, acts, emos):
        records.append({
            'dialogues': dial,
            'acts': act,
            'emos': emo
        })
    return records


def _translate_act(num: str):
    return _ACT_LOOKUP[num]


def _translate_emo(num: str):
    return _EMO_LOOKUP[num]


def _parse_utterances(conv: dict, conv_id: int):
    "split a conversation with associated acts and emos into generator of utterances. also tag with conv_id"
    utters = conv['dialogues'].strip('\n').split('__eou__')
    acts = conv['acts'].strip('\n ').split(' ')
    emos = conv['emos'].strip('\n ').split(' ')
    parsed_utters = []
    for utter_id, (utter, act, emo) in enumerate(zip(utters, acts, emos)):
        parsed_utters.append({
            'conv_id': conv_id,
            'utter_id': utter_id,
            'utterance': utter,
            'act': _translate_act(act),
            'emotion': _translate_emo(emo)
        })
    return parsed_utters


def _flatten(s: list):
    return [i for j in s for i in j]


def data():
    "get daily dialogue data (data frame friendly)"
    dialogues = _get_dialogues()
    parsed_dialogues = []
    for conv_id, dialogue in enumerate(dialogues):
        parsed_utterances = _parse_utterances(dialogue, conv_id)
        parsed_dialogues.append(parsed_utterances)
    return _flatten(parsed_dialogues)

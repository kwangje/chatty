import pandas as pd

from research.data import daily_dialogue
from research import utils


def test_train_test_split():
    data = daily_dialogue.data()
    df = pd.DataFrame(data)
    train, test = utils._make_train_test_split(df, 1000)
    assert test['conv_id'].unique().shape == (1000,)
    assert df['conv_id'].unique().shape[0] - train['conv_id'].unique().shape[0] == 1000


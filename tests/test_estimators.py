import pytest
import pandas as pd
from functools import partial
from string import ascii_letters
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from epic.sklearn.composite import ThresholdCompositeClassifier, WeightedLinearOVRClassifier
from epic.sklearn.preprocessing import (
    FrequencyTransformer, FrequencyListTransformer, ListStatisticsTransformer,
    ManyHotEncoder, BinningTransformer, YeoJohnsonTransformer,
    TailChopper, SimpleTransformer, DataFrameWrapper,
    DataFrameColumnSelector, FeatureGenerator,
    LabelBinarizerWithMissingValues, MultiLabelEncoder,
)

try:
    # Just check if sklearn version is at least 1.6
    from sklearn.utils.validation import validate_data
except ImportError:
    pass
else:
    parametrize_with_checks = partial(
        parametrize_with_checks,
        expected_failed_checks=lambda estimator: getattr(estimator, '_xfail_checks', dict)(),
    )


@parametrize_with_checks([
    ThresholdCompositeClassifier(LogisticRegression()),
    WeightedLinearOVRClassifier(LogisticRegression()),
    FrequencyTransformer(),
    FrequencyListTransformer(),
    ListStatisticsTransformer(),
    ManyHotEncoder(),
    BinningTransformer(),
    YeoJohnsonTransformer(),
    TailChopper(),
    SimpleTransformer(),
    DataFrameWrapper(LogisticRegression()),
    DataFrameColumnSelector(),
    FeatureGenerator(),
    LabelBinarizerWithMissingValues(),
    MultiLabelEncoder(),
])
def test_sklearn_compatibility(estimator, check):
    check(estimator)


@pytest.fixture
def dataframe(n=10) -> pd.DataFrame:
    return pd.DataFrame(
        data=dict(
            A=range(n),
            B=['bla'] * n,
            C=[[1, 2, 3]] * n
        ),
        index=list(ascii_letters[:n]),
    )


def transform_and_assert_dataframe(X, transformer, same_columns=True) -> pd.DataFrame:
    Xt = transformer.fit_transform(X)
    assert isinstance(Xt, pd.DataFrame)
    assert X.index.equals(Xt.index)
    if same_columns:
        assert X.columns.equals(Xt.columns)
    return Xt


def test_binning_transformer(dataframe):
    Xt = transform_and_assert_dataframe(
        dataframe,
        BinningTransformer(bins={'A': 2}),
    )
    half = len(dataframe) // 2
    assert (Xt['A'] == [0] * half + [1] * (len(dataframe) - half)).all()


def test_yeo_johnson_transformer(dataframe):
    transform_and_assert_dataframe(
        dataframe.select_dtypes('number'),
        YeoJohnsonTransformer(),
    )


def test_tail_chopper(dataframe):
    transform_and_assert_dataframe(
        dataframe.select_dtypes('number'),
        TailChopper(),
    )


def test_simple_transformer(dataframe):
    Xt = transform_and_assert_dataframe(
        dataframe,
        SimpleTransformer(),
    )
    assert Xt.equals(dataframe)


def test_dataframe_wrapper(dataframe):
    transform_and_assert_dataframe(
        dataframe.select_dtypes('number'),
        DataFrameWrapper(StandardScaler()),
    )


def test_dataframe_column_selector(dataframe):
    Xt = transform_and_assert_dataframe(
        dataframe,
        DataFrameColumnSelector(['A']),
        same_columns=False,
    )
    assert Xt.columns.tolist() == ['A']
    assert Xt.equals(dataframe[['A']])


def test_multi_label_encoder(dataframe):
    y = dataframe['B']
    yt = MultiLabelEncoder(allow_singles=True).fit_transform(y)
    assert isinstance(yt, pd.Series)
    assert yt.index.equals(y.index)
    assert yt.to_list() == [[0]] * len(y)


def test_frequency_transformer(dataframe):
    transform_and_assert_dataframe(
        dataframe[['A']],
        FrequencyTransformer(),
    )


def test_list_statistics_transformer(dataframe):
    transform_and_assert_dataframe(
        dataframe['C'],
        ListStatisticsTransformer(),
        same_columns=False,
    )


def test_many_hot_encoder(dataframe):
    transform_and_assert_dataframe(
        dataframe,
        ManyHotEncoder(
            categorical_features=['B', 'C'],
            df_out='single',
            allow_singles=True
        ),
        same_columns=False,
    )
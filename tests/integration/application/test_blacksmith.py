import time
from typing import Type, Tuple

import pandas as pd
import pytest
from pydantic.dataclasses import dataclass

from feature_forger.application.blacksmith import Blacksmith
from feature_forger.domain.entities.entity_model import EntityModel
from feature_forger.domain.entities.feature import Feature
from tests.common.entity_models.transaction import Transaction
from tests.common.datasets.bank import BankDataset
from tests.common.features import TransactionMonth, \
    TransactionRoundedAmountChangeDiffDoubled, \
    TransactionRoundedAmountChangeDiffSquared, \
    TransactionHalfRoundedAmountChange, TransactionRoundedAmountChangeDiff, \
    TransactionAmountChange, TransactionRoundedAmountChange
from tests.common.features.transaction_rounded_amount_change_doubled import \
    TransactionRoundedAmountChangeDoubled


class TestBlacksmith:

    @pytest.fixture()
    def sut(self) -> Blacksmith:
        return Blacksmith()

    def test_should_create_features_with_concurrent_paths(self,
                                    sut,
                                    data):
        # arrange
        original_columns = data.columns
        recipes = sut.build_recipes(
            dataset=BankDataset(data=data),
            features=[
                TransactionMonth(),
                TransactionRoundedAmountChangeDiffDoubled(),
                TransactionRoundedAmountChangeDiffSquared(),
                TransactionHalfRoundedAmountChange(),
                TransactionRoundedAmountChangeDoubled()
            ],
            concurrent_paths=True)
        # act
        result = sut.forge(recipe=recipes[0])
        # assert
        assert isinstance(result, pd.DataFrame)
        assert original_columns.isin(result).all()
        assert TransactionHalfRoundedAmountChange.col_name in result
        assert TransactionMonth.col_name in result
        assert TransactionRoundedAmountChangeDiff.col_name in result
        assert TransactionAmountChange.col_name in result
        assert TransactionHalfRoundedAmountChange.col_name in result
        assert TransactionRoundedAmountChangeDiffDoubled.col_name in result
        assert TransactionRoundedAmountChangeDiffSquared.col_name in result

    def test_should_create_features_with_single_path(self,
                                    sut,
                                    data):
        # arrange
        original_columns = data.columns
        recipes = sut.build_recipes(
            dataset=BankDataset(data=data),
            features=[
                TransactionMonth(),
                TransactionRoundedAmountChangeDiffDoubled(),
                TransactionRoundedAmountChangeDiffSquared(),
                TransactionHalfRoundedAmountChange()
            ],
            concurrent_paths=False)
        # act
        result = sut.forge(recipe=recipes[0])
        # assert
        assert isinstance(result, pd.DataFrame)
        assert original_columns.isin(result).all()
        assert TransactionHalfRoundedAmountChange.col_name in result
        assert TransactionMonth.col_name in result
        assert TransactionRoundedAmountChangeDiff.col_name in result
        assert TransactionAmountChange.col_name in result
        assert TransactionHalfRoundedAmountChange.col_name in result
        assert TransactionRoundedAmountChangeDiffDoubled.col_name in result
        assert TransactionRoundedAmountChangeDiffSquared.col_name in result

    def test_results_match_from_both_methods(self, data, sut):
        # arrange
        single_path_recipe = sut.build_recipes(
            dataset=BankDataset(data=data),
            features=[
                TransactionMonth(),
                TransactionRoundedAmountChangeDiffDoubled(),
                TransactionRoundedAmountChangeDiffSquared(),
                TransactionHalfRoundedAmountChange()
            ],
            concurrent_paths=False)[0]
        concurrent_path_recipe = sut.build_recipes(
            dataset=BankDataset(data=data),
            features=[
                TransactionMonth(),
                TransactionRoundedAmountChangeDiffDoubled(),
                TransactionRoundedAmountChangeDiffSquared(),
                TransactionHalfRoundedAmountChange()
            ],
            concurrent_paths=True)[0]
        sut.forge(recipe=concurrent_path_recipe)
        sut.forge(recipe=single_path_recipe)
        # act
        start = time.time()
        single_path_result = sut.forge(recipe=single_path_recipe)
        print(f"Single path recipe: {time.time() - start}")
        start = time.time()
        concurrent_path_result = sut.forge(recipe=concurrent_path_recipe)
        print(f"Concurrent path recipe: {time.time() - start}")
        start = time.time()
        data = TransactionMonth().table_compute_fn(data.copy())
        data = TransactionAmountChange().table_compute_fn(data)
        data = TransactionRoundedAmountChange().table_compute_fn(data)
        data = TransactionRoundedAmountChangeDiff().table_compute_fn(data)
        data = TransactionRoundedAmountChangeDiffDoubled().table_compute_fn(data)
        data = TransactionHalfRoundedAmountChange().table_compute_fn(data)
        data = data.apply(TransactionRoundedAmountChangeDiffSquared().row_compute_fn, axis=1)
        print(f"Raw pandas: {time.time() - start}")
        assert data.shape == single_path_result.shape == concurrent_path_result.shape
        cols = data.columns
        pd.testing.assert_frame_equal(single_path_result[cols], data, check_dtype=False)
        pd.testing.assert_frame_equal(concurrent_path_result[cols], data, check_dtype=False)

    def test_byo_features_by_feature_dataclass(self, sut, data):
        # arrange
        trn_month_feature = TransactionMonth()


        def row_level_function(row):
            row['byo_feature'] = 'byo'
            return row

        recipes = sut.build_recipes(
            dataset=BankDataset(data=data),
            features=[
                trn_month_feature,
                Feature(
                    col_name='byo_feature',
                    description='description for byo feature',
                    entity_model=Transaction,
                    dependencies=(trn_month_feature,),
                    feature_name="BYOFeature",
                    row_level_function=row_level_function
                )
            ],
            concurrent_paths=False)
        # act
        result = sut.forge(recipe=recipes[0])
        # assert
        assert 'byo_feature' in result
        assert (result['byo_feature'] == 'byo').all()

    def test_byo_features_by_feature_subclass(self, sut, data):
        # arrange
        trn_month_feature = TransactionMonth()

        @dataclass(frozen=True)
        class BYOFeature(Feature):
            col_name: str = "byo_feature"
            description: str = 'description for byo feature'
            entity_model: Type[EntityModel] = Transaction
            dependencies: Tuple[Feature] = (trn_month_feature,)

            def row_compute_fn(self, row):
                row['byo_feature'] = 'byo'
                return row

        recipes = sut.build_recipes(
            dataset=BankDataset(data=data),
            features=[
                trn_month_feature,
                BYOFeature()
            ],
            concurrent_paths=False)
        # act
        result = sut.forge(recipe=recipes[0])
        # assert
        assert 'byo_feature' in result
        assert (result['byo_feature'] == 'byo').all()



from datetime import datetime

from pydantic import Field

from feature_forger.domain.entities.entity_model import EntityModel


class Transaction(EntityModel):
    account_no: str = Field(alias="Account No")
    date: datetime = Field(alias="DATE")
    trn_details: str = Field(alias="TRANSACTION DETAILS")
    chq_no: float = Field(alias="CHQ.NO.")
    value_date: datetime = Field(alias="VALUE DATE")
    withdrawal_amount: float = Field(alias="WITHDRAWAL AMT")
    deposit_amount: float = Field(alias="DEPOSIT AMT")
    balance_amount: float = Field(alias="BALANCE AMT")
import datetime
from io import BytesIO
import polars as pl
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
import requests
from typing import List, Literal, Optional


def download_data(url: str) -> pl.DataFrame:
    response = requests.get(url)
    response.raise_for_status()
    return pl.read_parquet(BytesIO(response.content))


class BloodDonationValidator(BaseModel):
    date: datetime.date = Field(..., description="Date of blood donations")
    state: str = Field(..., description="State where blood donation took place")
    blood_type: List[Literal["a", "b", "ab", "o", "all"]] = Field(..., description="Blood type")
    donations: int = Field(ge=0, description="Number of donations")


def validate_data(df: pl.DataFrame) -> None:
    for row in df.iter_rows(named=True):        
        BloodDonationValidator(
            date=row['date'],
            state=row['state'],
            blood_type=[row['blood_type']],
            donations=row['donations']
        )
        
    print("All rows have been validated successfully.")
    
    no_duplicated = df.is_duplicated().sum()
    
    if no_duplicated != 0:
        raise ValueError(f"{no_duplicated} duplicated rows have been detected.")
    print("No duplicated rows detected.")

    
class DonationPredictionRequest(BaseModel):
    lag1: int = Field(..., ge=0, lt=10000, description="No. of donations 1 day ago.")
    lag2: int = Field(..., ge=0, lt=10000, description="No. of donations 2 day ago.")
    lag3: int = Field(..., ge=0, lt=10000, description="No. of donations 3 day ago.")
    lag4: int = Field(..., ge=0, lt=10000, description="No. of donations 4 day ago.")
    lag5: int = Field(..., ge=0, lt=10000, description="No. of donations 5 day ago.")
    lag6: int = Field(..., ge=0, lt=10000, description="No. of donations 6 day ago.")
    lag7: int = Field(..., ge=0, lt=10000, description="No. of donations 7 day ago.")
    nextday: str = Field(..., description="Next day to predict donations for (format: YYYYMMDD).")
    high_donation_holiday: Optional[int] = Field(0, description="1 if next day is Hari Malaysia, Hari Kebangsaan, Hari Pekerja, or Hari Wesak.")
    low_donation_holiday: Optional[int] = Field(0, description="1 if next day is Hari Peristiwa, Hari Raya Puasa or Hari Raya Qurban.")
    religion_or_culture_holiday: Optional[int] = Field(0, description="1 if next day is a religious or cultural holiday.")
    other_holiday: Optional[int] = Field(0, description="1 if next day is any other national public holiday.")
    
    model_config = {
        "extra": "forbid", # No extra fields allowed
    }
    
    @field_validator('nextday')
    @classmethod
    def validate_nextday(cls, value: str) -> str:
        if len(value) != 8:
            raise ValidationError("nextday must be in the format YYYYMMDD.")
        try:
            datetime.datetime.strptime(value, '%Y%m%d')
        except ValueError:
            raise ValidationError("nextday must be in the format YYYYMMDD.")
        return value
    
    @model_validator(mode='after')
    def validate_lags(self):
        lags = [self.lag1, self.lag2, self.lag3, self.lag4, self.lag5, self.lag6, self.lag7]
        if all(lag == lags[0] for lag in lags):
            raise ValidationError("All lag values cannot be the same.")
        return self

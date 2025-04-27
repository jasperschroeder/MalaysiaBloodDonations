import requests
import polars as pl
from io import BytesIO
import datetime
from typing import List, Literal
from pydantic import BaseModel, Field

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
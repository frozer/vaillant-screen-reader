"""Request/response models for the markup helper API."""

from typing import Literal, Optional

from pydantic import BaseModel


class ValuesIn(BaseModel):
    temperature: Optional[int] = None
    burn: Optional[bool] = None
    heating: Optional[bool] = None
    hotwater: Optional[bool] = None
    pump: Optional[bool] = None
    gasvalve: Optional[bool] = None


class ReviewIn(BaseModel):
    action: Literal['confirm', 'too_dark', 'off', 'update']
    values: Optional[ValuesIn] = None


def row_out(row, image_exists):
    return {
        'idx': row.idx,
        'filename': row.filename,
        'source_dir': row.source_dir,
        'status': row.status,
        'recognized': row.recognized.to_dict(),
        'current': row.current.to_dict(),
        'too_dark': row.too_dark,
        'confirmed': row.confirmed,
        'retrain_needed': row.retrain_needed,
        'image_exists': image_exists,
    }

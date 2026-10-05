from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class ResponseBase(BaseModel):
    code: int = 0
    message: str = "success"
    warnings: List[str] = []

class SyncParseData(BaseModel):
    task_id: str
    error: Optional[str] = None
    result: Dict[str, Any]
    metadata: Dict[str, Any]

class SyncParseResponse(ResponseBase):
    data: SyncParseData

class AsyncSubmitData(BaseModel):
    task_id: str
    status: str

class AsyncSubmitResponse(ResponseBase):
    data: AsyncSubmitData

class AsyncQueryData(BaseModel):
    record_id: Optional[int] = None
    task_id: str
    status: str
    file_name: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str]
    metadata: Dict[str, Any]

class AsyncQueryResponse(ResponseBase):
    data: AsyncQueryData

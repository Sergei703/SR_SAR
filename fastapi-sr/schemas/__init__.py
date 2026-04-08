from pydantic import BaseModel
from typing import List


class ProcessingOptions(BaseModel):
    fast_preview: bool = False
    enable_subsampling: bool = True


class FileInfo(BaseModel):
    filename: str
    original_width: int
    original_height: int
    output_width: int
    output_height: int


class TaskInfo(BaseModel):
    task_id: str
    files: List[FileInfo]
    options: ProcessingOptions


class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: int = 0

from pydantic import BaseModel


class UserAPIKeyCreateRequest(BaseModel):
    name: str

    class Config:
        from_attributes = True


class UserAPIKeyCreateResponse(BaseModel):
    id: int
    api_key: str
    name: str

    class Config:
        from_attributes = True


class UserAPIKeyDeleteResponse(BaseModel):
    message: str

    class Config:
        from_attributes = True


class UserAPIKeyResponse(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True

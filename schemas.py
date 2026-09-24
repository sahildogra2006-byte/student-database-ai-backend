from pydantic import BaseModel, EmailStr, Field

class StudentCreate(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    age: int = Field(ge=1, le=100)
    course: str = Field(min_length=2, max_length=100)
    year: int = Field(ge=1, le=6)

class StudentUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=100)
    email: EmailStr | None = None
    age: int | None = Field(default=None, ge=1, le=100)
    course: str | None = Field(default=None, min_length=2, max_length=100)
    year: int | None = Field(default=None, ge=1, le=6)

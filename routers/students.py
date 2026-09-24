from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from database import get_session
from schemas import StudentCreate, StudentUpdate
from crud import create_student, get_students, get_student, update_student, delete_student

router = APIRouter(prefix="/students", tags=["Students"])

@router.post("/", status_code=201)
def create(data: StudentCreate, session: Session = Depends(get_session)):
    return create_student(session, data)

@router.get("/")
def read_all(session: Session = Depends(get_session)):
    return get_students(session)

@router.get("/{student_id}")
def read_one(student_id: int, session: Session = Depends(get_session)):
    student = get_student(session, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student

@router.put("/{student_id}")
def update(student_id: int, data: StudentUpdate, session: Session = Depends(get_session)):
    student = get_student(session, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return update_student(session, student, data)

@router.delete("/{student_id}")
def delete(student_id: int, session: Session = Depends(get_session)):
    student = get_student(session, student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    delete_student(session, student)
    return {"message": "Student deleted successfully"}

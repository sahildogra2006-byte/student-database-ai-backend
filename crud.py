from sqlmodel import Session, select
from models import Student
from schemas import StudentCreate, StudentUpdate

def create_student(session: Session, data: StudentCreate):
    student = Student(**data.model_dump())
    session.add(student)
    session.commit()
    session.refresh(student)
    return student

def get_students(session: Session):
    return session.exec(select(Student)).all()

def get_student(session: Session, student_id: int):
    return session.get(Student, student_id)

def update_student(session: Session, student: Student, data: StudentUpdate):
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(student, key, value)
    session.add(student)
    session.commit()
    session.refresh(student)
    return student

def delete_student(session: Session, student: Student):
    session.delete(student)
    session.commit()

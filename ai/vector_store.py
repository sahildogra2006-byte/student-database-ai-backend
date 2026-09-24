from qdrant_client import QdrantClient, models

from database import engine
from models import Student
from sqlmodel import Session, select


# Local Qdrant database
client = QdrantClient(path="qdrant_data")

# Name of our vector collection
COLLECTION_NAME = "student_information"

# Embedding model
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def create_collection():
    """Create the student vector collection if it doesn't exist."""

    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=client.get_embedding_size(MODEL_NAME),
            distance=models.Distance.COSINE,
        ),
    )

    print("Qdrant collection created successfully.")


def add_student(
    student_id,
    name,
    email,
    age,
    course,
    year,
):
    """Add one student to the vector database."""

    student_text = (
        f"Student ID: {student_id}. "
        f"Name: {name}. "
        f"Email: {email}. "
        f"Age: {age}. "
        f"Course: {course}. "
        f"Year: {year}."
    )

    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=[
            models.Document(
                text=student_text,
                model=MODEL_NAME,
            )
        ],
        payload=[
            {
                "student_id": student_id,
                "name": name,
                "email": email,
                "age": age,
                "course": course,
                "year": year,
                "text": student_text,
            }
        ],
        ids=[student_id],
    )

    print(f"Student {student_id} added to Qdrant.")


def sync_students_from_database():
    """Read all students from SQLite and add them to Qdrant."""

    create_collection()

    with Session(engine) as session:
        students = session.exec(select(Student)).all()

        for student in students:
            add_student(
                student.id,
                student.name,
                student.email,
                student.age,
                student.course,
                student.year,
            )

    print(f"Synced {len(students)} students to Qdrant.")


def search_students(question, limit=3):
    """Search students using the meaning of the question."""

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.Document(
            text=question,
            model=MODEL_NAME,
        ),
        limit=limit,
        with_payload=True,
    ).points

    return results


if __name__ == "__main__":
    sync_students_from_database()

    print("Qdrant vector store is ready.")
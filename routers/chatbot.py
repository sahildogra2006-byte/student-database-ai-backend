from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session

from database import get_session
from ai.chatbot import answer_question

router = APIRouter(
    prefix="/chatbot",
    tags=["AI Chatbot"]
)


class ChatRequest(BaseModel):
    question: str


@router.post("/ask")
def ask(
    request: ChatRequest,
    session: Session = Depends(get_session)
):
    return answer_question(request.question, session)
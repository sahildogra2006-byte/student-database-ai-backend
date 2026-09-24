import json
import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from google import genai
from pydantic import BaseModel
from sqlmodel import Session, select
from langgraph.graph import StateGraph, START, END

from database import get_session
from models import Student
from ai.vector_store import search_students


load_dotenv()


router = APIRouter(
    prefix="/chatbot",
    tags=["Chatbot"]
)


# =========================================================
# REQUEST MODEL
# =========================================================

class ChatRequest(BaseModel):
    question: str


# =========================================================
# CHAT STATE
# =========================================================

class ChatState(TypedDict):
    question: str
    intent: str
    student_name: str
    student_id: int | None
    course: str
    year: int | None
    age: int | None
    comparison: str
    fields: list[str]
    answer: str
    session: Session
    vector_results: list[dict]


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# EXTRACT STUDENT ID
# =========================================================
#
# Supports:
# "ID 1"
# "ID: 1"
# "ID = 1"
# "ID is 1"
# "student ID 1"
# "student ID is 1"
# "student whose ID is 1"
# "student whose id = 1"
# "student with ID 1"
#
# =========================================================

def extract_student_id(question: str):

    patterns = [

        # "student whose ID is 1"
        r"\bstudent\s+whose\s+id\s*(?:is|=|:|#)?\s*(\d+)\b",

        # "student with ID 1"
        r"\bstudent\s+with\s+id\s*(?:is|=|:|#)?\s*(\d+)\b",

        # "student ID 1", "student ID is 1"
        r"\bstudent\s+id\s*(?:is|=|:|#)?\s*(\d+)\b",

        # "ID 1", "ID: 1", "ID = 1", "ID is 1"
        r"\bid\s*(?:is|=|:|#)?\s*(\d+)\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            re.IGNORECASE
        )

        if match:
            return int(
                match.group(1)
            )

    return None


# =========================================================
# EXTRACT EMAIL
# =========================================================

def extract_email_from_question(question: str):

    match = re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        question
    )

    if match:
        return match.group(0)

    return ""


# =========================================================
# EMAIL VERIFICATION QUESTION
# =========================================================

def is_email_verification_question(question: str):

    q = question.casefold()

    expected_email = extract_email_from_question(
        question
    )

    # An email address must be present for this
    # to be an email verification question.
    if not expected_email:
        return False

    verification_patterns = [

        # Does student ID 1 have the email user@gmail.com?
        r"\bdoes\b.*\bemail\b",

        # Has student ID 1 the email...
        r"\bhas\b.*\bemail\b",

        # Have student ID 1 the email...
        r"\bhave\b.*\bemail\b",

        # Is the email ... correct/right?
        r"\bis\b.*\bemail\b.*\b(?:correct|right)\b",

        # NEW:
        # Is user@gmail.com the email of student ID 1?
        #
        # The email is already required above, so this
        # safely identifies verification-style questions.
        r"\bis\b.*\b(?:the\s+)?email\b",

        # Email ... correct/right/match
        r"\bemail\b.*\b(?:correct|right|match)\b",

        # Email ... same/valid
        r"\bemail\b.*\b(?:same|valid)\b",

        # Email belongs to student...
        r"\bemail\b.*\bbelongs?\b",

        # Email of student...
        r"\bemail\b.*\bof\b.*\bstudent\b"
    ]

    return any(
        re.search(
            pattern,
            q
        )
        for pattern in verification_patterns
    )


# =========================================================
# ALL STUDENTS NAME + COURSE QUESTION
# =========================================================

def is_students_name_course_list_question(
    question: str
):

    q = question.casefold()

    has_name = bool(
        re.search(
            r"\bnames?\b",
            q
        )
    )

    has_course = (
        "course" in q
        or "courses" in q
    )

    has_multiple_students = bool(
        re.search(
            r"\bstudents?\b",
            q
        )
    )

    has_list_word = any(
        phrase in q
        for phrase in [
            "show me",
            "show",
            "list",
            "give me",
            "tell me",
            "all students",
            "everyone"
        ]
    )

    # NEW:
    # Support questions that don't explicitly contain
    # the word "names".
    #
    # Examples:
    # "List all students with their courses."
    # "Show me all students and their courses."
    # "Give me all students with courses."
    # "Show students and courses."
    has_students_with_courses = (
        "students with their courses" in q
        or "students and their courses" in q
        or "all students with their courses" in q
        or "all students and their courses" in q
        or "students with courses" in q
        or "students and courses" in q
        or "all students with courses" in q
        or "all students and courses" in q
    )

    return (
        has_course
        and has_multiple_students
        and has_list_word
        and (
            has_name
            or has_students_with_courses
        )
    )


# =========================================================
# BROAD COMPUTER SCIENCE SEMANTIC QUESTION
# =========================================================

def is_broad_computer_science_question(
    question: str
):

    q = question.casefold()

    has_topic = (
        "computer science" in q
        or bool(
            re.search(
                r"\bcse\b",
                q
            )
        )
    )

    if not has_topic:
        return False

    has_study_word = bool(
        re.search(
            r"\b(?:study|studies|studying)\b",
            q
        )
    )

    if not has_study_word:
        return False

    broad_words = [
        "everyone",
        "all students",
        "which students",
        "find all students",
        "find students",
        "tell me everyone",
        "show me everyone",
        "list students",
        "show students",
        "tell me which students",
        "i want to find",
        "can you find"
    ]

    return any(
        phrase in q
        for phrase in broad_words
    )


# =========================================================
# NEW: EXTRACT EXACT COURSE COUNT REQUEST
# =========================================================
#
# Examples:
#
# "How many BTech CSE students are there?"
# -> BTech CSE
#
# "How many BTech - CSE students are there?"
# -> BTech - CSE
#
# This keeps BTech CSE and BTech - CSE separate.
#
# =========================================================

def extract_course_count_request(question: str):

    q = question.strip()

    patterns = [

        r"\bhow\s+many\s+(.+?)\s+students?\s+are\s+there\b",

        r"\bhow\s+many\s+(.+?)\s+students?\s+do\s+we\s+have\b",

        r"\bhow\s+many\s+(.+?)\s+students?\s+are\s+enrolled\b",

        r"\bcount\s+(?:the\s+)?(.+?)\s+students?\b",

        r"\bnumber\s+of\s+(.+?)\s+students?\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            q,
            re.IGNORECASE
        )

        if not match:
            continue

        course = match.group(1).strip(
            " ?.,"
        )

        course = re.sub(
            r"^(?:the|all)\s+",
            "",
            course,
            flags=re.IGNORECASE
        ).strip()

        if not course:
            continue

        return course

    return ""


# =========================================================
# EXTRACT STUDENT NAME
# =========================================================

def extract_student_name(question: str):

    q = question.strip()

    patterns = [

        r"\b(?:what\s+course\s+is|what\s+is)\s+([A-Za-z]+(?:\s+[A-Za-z]+){1,2})\s+(?:studying|age|email|course|year)\b",

        r"\b(?:is|does)\s+([A-Za-z]+(?:\s+[A-Za-z]+){1,2})\s+(?:studying|in)\b",

        r"\b(?:tell\s+me|show\s+me|give\s+me)\s+([A-Za-z]+(?:\s+[A-Za-z]+){1,2})\s+(?:age|email|course|year)\b",

        r"\bof\s+([A-Za-z]+(?:\s+[A-Za-z]+){1,3})\s*[?.!,]?\s*$",

        r"\b([A-Za-z]+(?:\s+[A-Za-z]+){1,3})['’]s\b",

        r"\bfor\s+([A-Za-z]+(?:\s+[A-Za-z]+){1,3})\s*[?.!,]?\s*$",

        r"\bstudent\s+([A-Za-z]+(?:\s+[A-Za-z]+){1,3})\s*[?.!,]?\s*$",

        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*[?.!,]?\s*$"
    ]

    extra_patterns = [

        r"\bwhat(?:'s|\s+is)?\s+([A-Za-z]+(?:\s+[A-Za-z]+){1,3})\s+(?:age|email|course|year)\b",

        r"^([A-Za-z]+(?:\s+[A-Za-z]+){1,3})\s+(?:age|email|course|year)\s*[?.!,]?\s*$"
    ]

    patterns = extra_patterns + patterns

    ignored_words = {
        "student",
        "details",
        "detail",
        "information",
        "info",
        "course",
        "age",
        "email",
        "year",
        "id",
        "name"
    }

    for pattern in patterns:

        match = re.search(
            pattern,
            q,
            re.IGNORECASE
        )

        if not match:
            continue

        name = match.group(1).strip()

        name = name.rstrip(
            "?.!,:"
        )

        words = name.split()

        if not words:
            continue

        if any(
            word.lower() in ignored_words
            for word in words
        ):
            continue

        return name

    return ""


# =========================================================
# EXTRACT REQUESTED FIELDS
# =========================================================

def extract_requested_fields(question: str):

    q = question.lower()

    fields = []

    # =====================================================
    # ID
    # =====================================================

    explicit_id_request = (
        "what is the id" in q
        or "what's the id" in q
        or "give me the id" in q
        or "show me the id" in q
        or "tell me the id" in q
        or "student id" in q
        or "id of the student" in q
        or "id of student" in q
    )

    if explicit_id_request:
        fields.append("id")

    # =====================================================
    # NAME
    # =====================================================

    if (
        "name" in q
        or "student name" in q
    ):
        fields.append("name")

    # =====================================================
    # EMAIL
    # =====================================================

    if (
        "email" in q
        or "email address" in q
    ):
        fields.append("email")

    # =====================================================
    # AGE
    # =====================================================

    if (
        "age" in q
        or re.search(r"\bhow\s+old\b", q)
    ):
        fields.append("age")

    # =====================================================
    # COURSE
    # =====================================================

    if (
        "course" in q
        or "studying" in q
    ):
        fields.append("course")

    # =====================================================
    # YEAR
    # =====================================================

    if (
        "year" in q
        or "class" in q
    ):
        fields.append("year")

    # =====================================================
    # GENERAL INFORMATION / DETAILS
    # =====================================================

    if (
        "information about" in q
        or "info about" in q
        or "details about" in q
        or "information on" in q
        or "details on" in q
    ):

        return [
            "name",
            "id",
            "email",
            "age",
            "course",
            "year"
        ]

    # =====================================================
    # ALL DETAILS
    # =====================================================

    if (
        "all details" in q
        or "complete details" in q
        or "full details" in q
        or "all information" in q
        or "complete information" in q
        or "everything about" in q
    ):

        return [
            "name",
            "id",
            "email",
            "age",
            "course",
            "year"
        ]

    return list(
        dict.fromkeys(fields)
    )


# =========================================================
# FIND STUDENT NAME IN DATABASE
# =========================================================

def find_student_name_in_question(
    session: Session,
    question: str
):

    q = question.casefold()

    students = session.exec(
        select(Student)
    ).all()

    matches = [
        student.name
        for student in students
        if student.name
        and student.name.casefold() in q
    ]

    if not matches:
        return ""

    return max(
        matches,
        key=len
    )


# =========================================================
# COURSE MATCHING HELPER
# =========================================================

def normalize_course_for_matching(course: str):

    value = str(course or "").strip().casefold()

    # Normalize punctuation/spacing so values such as:
    # BTech CSE, BTech - CSE, B.Tech CSE and B.Tech - CSE
    # can be compared consistently.
    value = re.sub(r"\bb\.?tech\b", "btech", value)
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()

    # For matching purposes, a BTech specialization and its bare
    # specialization are treated as the same course. This makes
    # the matching dynamic for future courses such as ECE, IT,
    # AIML, ME, etc., instead of hard-coding individual courses.
    if value.startswith("btech "):
        value = value[6:].strip()

    return value


def course_matches_requested(
    actual_course: str,
    requested_course: str
):

    actual = normalize_course_for_matching(
        actual_course
    )

    requested = normalize_course_for_matching(
        requested_course
    )

    # BTech means any BTech course.
    if str(requested_course or "").strip().casefold() == "btech":
        return str(actual_course or "").strip().casefold().startswith("btech")

    return actual == requested


def find_course_in_question(
    session: Session,
    question: str
):

    q = question.casefold()

    courses = session.exec(
        select(Student.course)
    ).all()

    # Remove duplicates and empty values, then check the longest
    # database course names first. This lets the database itself
    # define which courses the chatbot can recognize.
    unique_courses = list(dict.fromkeys(
        str(course).strip()
        for course in courses
        if course and str(course).strip()
    ))

    unique_courses.sort(
        key=lambda value: len(value),
        reverse=True
    )

    for course in unique_courses:

        normalized_course = normalize_course_for_matching(course)

        if not normalized_course:
            continue

        # Direct normalized course phrase in the question.
        if normalized_course in normalize_course_for_matching(q):
            return course

        # Also recognize a bare specialization such as "AIML"
        # when the stored course is "BTech AIML".
        if re.fullmatch(
            rf".*\b{re.escape(normalized_course)}\b.*",
            normalize_course_for_matching(q)
        ):
            return course

    return ""


# =========================================================
# GEMINI QUESTION UNDERSTANDING
# =========================================================

def understand_question(
    question: str,
    session: Session
):

    q = question.lower().strip()

    # =====================================================
    # EARLY UNSUPPORTED STUDENT FIELD OVERRIDE
    # =====================================================
    # The Student model only contains: id, name, email, age, course,
    # and year. Questions about fields such as salary or phone number
    # must not fall through to the generic ID lookup.
    # =====================================================

    unsupported_field_patterns = {
        "salary": r"\bsalar(?:y|ies)\b",
        "phone number": r"\b(?:phone|mobile|contact)(?:\s+number|\s+no\.?|)\b",
        "address": r"\baddress\b",
        "cgpa": r"\bcgpa\b",
        "marks": r"\bmarks?\b",
    }

    for unsupported_field, pattern in unsupported_field_patterns.items():

        if re.search(pattern, q, re.IGNORECASE):

            return {
                "intent": "unsupported_student_field",
                "student_id": None,
                "student_name": "",
                "course": "",
                "year": None,
                "age": None,
                "comparison": "",
                "fields": [],
                "unsupported_field": unsupported_field
            }

    # =====================================================
    # EARLY EXACT AGE LOOKUP OVERRIDE
    # =====================================================
    # Handle natural wording such as:
    # "Which student is 19 years old?"
    # "Who is 19 years old?"
    # "Which students are 19 years old?"
    # This runs before the existing semantic/Gemini logic so
    # the question cannot be misclassified as a student-name
    # lookup or semantic search.
    # =====================================================

    early_exact_age_match = re.search(
        r"\b(?:which\s+student|which\s+students|who)\s+(?:is|are)\s+(\d+)\s+years?\s+old\b",
        q
    )

    if early_exact_age_match:

        exact_age = int(
            early_exact_age_match.group(1)
        )

        return {
            "intent": "age_exact",
            "student_id": None,
            "student_name": "",
            "course": "",
            "year": None,
            "age": exact_age,
            "comparison": "exact",
            "fields": []
        }

    # =====================================================
    # NORMALIZE YEAR WORDS
    # =====================================================

    year_words = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4
    }

    for word, number in year_words.items():

        q = re.sub(
            rf"\b{word}\s+year\b",
            f"{number} year",
            q
        )

    # =====================================================
    # NORMALIZE HYPHENATED YEAR WORDS
    # =====================================================

    for word, number in year_words.items():

        q = re.sub(
            rf"\b{word}[-\s]+year\b",
            f"{number} year",
            q
        )

    # =====================================================
    # DIRECT NAME-ALL-STUDENTS-BY-COURSE OVERRIDE
    # =====================================================
    # Handle natural wording such as:
    # "name all the student in BTech CSE"
    # "name all students in BTech - CSE"
    # "name the students in BTech CSE"
    # This must run BEFORE the existing Gemini logic so the
    # question cannot be misclassified as a single-student lookup.
    # =====================================================

    direct_course = find_course_in_question(
        session,
        question
    )

    if direct_course and (
        re.search(r"\bname(?:s)?\b", q, re.IGNORECASE)
        and re.search(r"\bstudents?\b", q, re.IGNORECASE)
        and re.search(r"\b(?:in|from|studying)\b", q, re.IGNORECASE)
    ):

        return {
            "intent": "students_by_course",
            "student_id": None,
            "student_name": "",
            "course": direct_course,
            "year": None,
            "age": None,
            "comparison": "",
            "fields": ["name"]
        }

    # =====================================================
    # NAME + COURSE LIST
    # =====================================================

    if is_students_name_course_list_question(
        question
    ):

        return {
            "intent": "students_fields_list",
            "student_id": None,
            "student_name": "",
            "course": "",
            "year": None,
            "age": None,
            "comparison": "",
            "fields": [
                "name",
                "course"
            ]
        }

    # =====================================================
    # EMAIL VERIFICATION
    # =====================================================

    detected_email = extract_email_from_question(
        question
    )

    detected_id_for_email = extract_student_id(
        question
    )

    if (
        detected_id_for_email is not None
        and detected_email
        and is_email_verification_question(
            question
        )
    ):

        return {
            "intent": "verify_email",
            "student_id": detected_id_for_email,
            "student_name": "",
            "course": "",
            "year": None,
            "age": None,
            "comparison": "",
            "fields": []
        }

    # =====================================================
    # EXACT COURSE COUNT
    # =====================================================
    #
    # IMPORTANT:
    # This is before the generic course logic so:
    #
    # BTech CSE
    # BTech - CSE
    #
    # remain separate courses.
    #
    # =====================================================

    detected_course_count = extract_course_count_request(
        question
    )

    if detected_course_count:

        normalized_course = detected_course_count

        if re.fullmatch(
            r"btech\s*[-–—]\s*cse",
            normalized_course,
            re.IGNORECASE
        ):

            normalized_course = "BTech - CSE"

        elif re.fullmatch(
            r"btech\s+cse",
            normalized_course,
            re.IGNORECASE
        ):

            normalized_course = "BTech CSE"

        elif re.fullmatch(
            r"aiml",
            normalized_course,
            re.IGNORECASE
        ):

            normalized_course = "AIML"

        elif re.fullmatch(
            r"btech",
            normalized_course,
            re.IGNORECASE
        ):

            normalized_course = "BTech"

        return {
            "intent": "count_by_course",
            "student_id": None,
            "student_name": "",
            "course": normalized_course,
            "year": None,
            "age": None,
            "comparison": "",
            "fields": []
        }

    # =====================================================
    # COUNT BY YEAR
    # =====================================================

    count_year_match = re.search(
        r"\bhow many\b.*?\b([1-4])(?:st|nd|rd|th)?\s*year\b",
        q
    )

    if count_year_match:

        return {
            "intent": "count_by_year",
            "student_id": None,
            "student_name": "",
            "course": "",
            "year": int(
                count_year_match.group(1)
            ),
            "age": None,
            "comparison": "",
            "fields": []
        }

    # =====================================================
    # STUDENTS BY YEAR
    # =====================================================

    year_list_patterns = [

        r"\bwho\s+(?:is|are)\s+in\s+(?:the\s+)?([1-4])(?:st|nd|rd|th)?\s*year\b",

        r"\bwho\s+(?:studies|study|is\s+studying|are\s+studying)\s+in\s+(?:the\s+)?([1-4])(?:st|nd|rd|th)?\s*year\b",

        r"\b(?:tell|show|list|give)\s+(?:me\s+)?(?:the\s+)?students?\s+(?:are\s+)?in\s+([1-4])(?:st|nd|rd|th)?\s*year\b",

        r"\b(?:tell|show|list|give)\s+(?:me\s+)?(?:the\s+)?students?\s+(?:from|of)\s+(?:the\s+)?([1-4])(?:st|nd|rd|th)?\s*year\b",

        r"\bstudents?\s+(?:are\s+)?in\s+(?:the\s+)?([1-4])(?:st|nd|rd|th)?\s*year\b",

        r"\blist\s+(?:all\s+)?(?:the\s+)?([1-4])\s*year\s+students?\b",

        r"\b(?:tell|show|give)\s+(?:me\s+)?(?:everyone|all\s+students?)\s+(?:in|from)\s+(?:the\s+)?([1-4])\s*year\b",

        r"\bwho\s+studies\s+in\s+(?:the\s+)?year\s+([1-4])\b",

        r"\bwho\s+is\s+in\s+(?:the\s+)?year\s+([1-4])\b",

        r"\bwho\s+(?:is|are)\s+in\s+(?:the\s+)?([1-4])\s*year\b",

        r"\bwhich\s+students?\s+(?:belong\s+to|are\s+in)\s+(?:the\s+)?([1-4])\s*year\b",

        r"\b(?:show|list|give|tell)\s+(?:me\s+)?(?:the\s+)?([1-4])\s*year\s+students?\b",

        r"\bi\s+want\s+(?:a\s+)?list\s+of\s+(?:the\s+)?([1-4])\s*year\s+students?\b"
    ]

    for pattern in year_list_patterns:

        year_list_match = re.search(
            pattern,
            q
        )

        if year_list_match:

            detected_year = int(
                year_list_match.group(1)
            )

            return {
                "intent": "students_by_year",
                "student_id": None,
                "student_name": "",
                "course": "",
                "year": detected_year,
                "age": None,
                "comparison": "",
                "fields": []
            }

    # =====================================================
    # DETERMINISTIC COURSE FILTERS
    # =====================================================

    course = find_course_in_question(
        session,
        question
    )

    if course and (
        "which students" in q
        or "which student" in q
        or "students studying" in q
        or "students are studying" in q
        or "everyone" in q
        or "all students" in q
        or "any students" in q
        or "are there any" in q
        or "find students" in q
        or "list students" in q
        or "tell me everyone" in q
        or "i want a list" in q
        or re.search(
            r"\bwho\s+(?:is|are)\s+(?:studying|in)\b",
            q
        )
        or re.search(
            r"\bwho\s+studies\b",
            q
        )
    ):

        return {
            "intent": "students_by_course",
            "student_id": None,
            "student_name": "",
            "course": course,
            "year": None,
            "age": None,
            "comparison": "",
            "fields": []
        }

    # =====================================================
    # YEAR EXTRACTION
    # =====================================================

    year = None

    year_match = re.search(
        r"\b([1-4])(?:st|nd|rd|th)?\s*year\b",
        q
    )

    if year_match:

        year = int(
            year_match.group(1)
        )

    if year is None:

        year_number_match = re.search(
            r"\byear\s+([1-4])\b",
            q
        )

        if year_number_match:

            year = int(
                year_number_match.group(1)
            )

    # =====================================================
    # DETERMINISTIC STUDENT NAME
    # =====================================================

    deterministic_name = find_student_name_in_question(
        session,
        question
    )

    # =====================================================
    # AIML COURSE FILTER
    # =====================================================

    aiml_course_match = re.search(
        r"\baiml\b",
        q,
        re.IGNORECASE
    )

    if aiml_course_match and (
        "which students" in q
        or "which student" in q
        or "students studying" in q
        or "students are studying" in q
        or "everyone" in q
        or "all students" in q
        or "any students" in q
        or "are there any" in q
        or "find students" in q
        or "list students" in q
        or "tell me everyone" in q
        or "i want a list" in q
        or re.search(
            r"\bwho\s+(?:is|are)\s+(?:studying|in)\b",
            q
        )
        or re.search(
            r"\bwho\s+studies\b",
            q
        )
    ):

        return {
            "intent": "students_by_course",
            "student_id": None,
            "student_name": "",
            "course": "AIML",
            "year": None,
            "age": None,
            "comparison": "",
            "fields": []
        }

    # =====================================================
    # GENERAL YEAR QUESTIONS
    # =====================================================

    if year is not None:

        year_question_words = [
            "who",
            "which",
            "students",
            "student",
            "list",
            "show",
            "tell",
            "everyone",
            "any",
            "find",
            "give me",
            "i want",
            "studies",
            "study",
            "studying",
            "belong"
        ]

        is_year_student_question = any(
            word in q
            for word in year_question_words
        )

        is_count_question = (
            "how many" in q
            or "number of" in q
            or "count" in q
        )

        if is_count_question:

            return {
                "intent": "count_by_year",
                "student_id": None,
                "student_name": "",
                "course": "",
                "year": year,
                "age": None,
                "comparison": "",
                "fields": []
            }

        if is_year_student_question:

            return {
                "intent": "students_by_year",
                "student_id": None,
                "student_name": "",
                "course": "",
                "year": year,
                "age": None,
                "comparison": "",
                "fields": []
            }

    # =====================================================
    # STUDENT ID LOOKUP
    # =====================================================

    student_id = extract_student_id(
        question
    )

    if student_id is not None:

        expected_email = extract_email_from_question(
            question
        )

        if (
            expected_email
            and is_email_verification_question(
                question
            )
        ):

            return {
                "intent": "verify_email",
                "student_id": student_id,
                "student_name": "",
                "course": "",
                "year": None,
                "age": None,
                "comparison": "",
                "fields": []
            }

        id_fields = extract_requested_fields(
            question
        )

        # If the question is ONLY asking for a student ID reference
        # (for example, "Student ID 1"), treat that ID as the way
        # to find the student and return all student details.
        q_clean = question.strip().casefold()

        id_only_patterns = [
            r"^student\s+id\s*(?:is|=|:|#)?\s*\d+\s*[?.!]*$",
            r"^id\s*(?:is|=|:|#)?\s*\d+\s*[?.!]*$",
            r"^student\s+whose\s+id\s*(?:is|=|:|#)?\s*\d+\s*[?.!]*$",
            r"^student\s+with\s+id\s*(?:is|=|:|#)?\s*\d+\s*[?.!]*$"
        ]

        is_id_only_question = any(
            re.fullmatch(pattern, q_clean)
            for pattern in id_only_patterns
        )

        if is_id_only_question:
            id_fields = [
                "name",
                "id",
                "email",
                "age",
                "course",
                "year"
            ]

        if not id_fields:

            id_fields = [
                "name",
                "id",
                "email",
                "age",
                "course",
                "year"
            ]

        return {
            "intent": "student_fields",
            "student_id": student_id,
            "student_name": "",
            "course": "",
            "year": None,
            "age": None,
            "comparison": "",
            "fields": id_fields
        }

    # =====================================================
    # SEMANTIC SEARCH
    # =====================================================

    generic_semantic_patterns = [

        r"\bwho\s+is\s+studying\s+computer\s+science\b",

        r"\bwho\s+are\s+studying\s+computer\s+science\b",

        r"\bwho\s+studies\s+computer\s+science\b",

        r"\bwhich\s+students\s+study\s+computer\s+science\b",

        r"\bwhich\s+students\s+are\s+studying\s+computer\s+science\b",

        r"\bshow\s+me\s+students\s+studying\s+computer\s+science\b",

        r"\bshow\s+me\s+students\s+studying\s+cse\b",

        r"\bwho\s+studies\s+cse\b",

        r"\bwho\s+is\s+studying\s+cse\b",

        r"\bwhich\s+students\s+study\s+cse\b",

        r"\bwhich\s+students\s+are\s+studying\s+cse\b",

        r"\bcan\s+you\s+find\s+students\s+studying\s+computer\s+science\b",

        r"\bfind\s+students\s+studying\s+computer\s+science\b",

        r"\blist\s+students\s+studying\s+computer\s+science\b",

        r"\btell\s+me\s+which\s+students\s+are\s+studying\s+computer\s+science\b",

        r"\bcan\s+you\s+find\s+students\s+studying\s+cse\b",

        r"\bfind\s+students\s+studying\s+cse\b",

        r"\blist\s+students\s+studying\s+cse\b",

        r"\bwhich\s+students\s+are\s+related\s+to\s+computer\s+science\b",

        r"\bwhich\s+students\s+are\s+related\s+to\s+cse\b",

        r"\btell\s+me\s+everyone\s+studying\s+computer\s+science\b",

        r"\btell\s+me\s+everyone\s+who\s+studies\s+computer\s+science\b",

        r"\btell\s+me\s+everyone\s+studying\s+cse\b",

        r"\bi\s+want\s+to\s+find\s+all\s+students\s+who\s+study\s+computer\s+science\b",

        r"\bi\s+want\s+to\s+find\s+all\s+students\s+who\s+study\s+cse\b",

        r"\bfind\s+all\s+students\s+who\s+study\s+computer\s+science\b",

        r"\bfind\s+all\s+students\s+who\s+study\s+cse\b"
    ]

    if is_broad_computer_science_question(
        question
    ):

        return {
            "intent": "semantic_search",
            "student_id": None,
            "student_name": "",
            "course": "",
            "year": None,
            "age": None,
            "comparison": "",
            "fields": []
        }

    if any(
        re.search(
            pattern,
            q
        )
        for pattern in generic_semantic_patterns
    ):

        return {
            "intent": "semantic_search",
            "student_id": None,
            "student_name": "",
            "course": "",
            "year": None,
            "age": None,
            "comparison": "",
            "fields": []
        }

    # =====================================================
    # STUDENT LOOKUP
    # =====================================================

    student_name = (
        deterministic_name
        or extract_student_name(question)
    )

    fields = extract_requested_fields(
        question
    )

    # =====================================================
    # NATURAL AGE QUESTION
    # =====================================================

    if (
        deterministic_name
        and re.search(
            r"\bhow\s+old\s+is\b",
            q
        )
    ):

        return {
            "intent": "student_fields",
            "student_id": None,
            "student_name": deterministic_name,
            "course": course,
            "year": year,
            "age": None,
            "comparison": "",
            "fields": ["age"]
        }

    # =====================================================
    # GENERAL STUDENT REQUEST
    # =====================================================

    if deterministic_name and (
        re.search(
            r"\b(?:tell|show|give)\s+me\s+about\b",
            q
        )
        or "details" in q
        or "information" in q
        or re.search(
            r"\binfo\b",
            q
        )
    ):

        return {
            "intent": "student_fields",
            "student_id": None,
            "student_name": deterministic_name,
            "course": course,
            "year": year,
            "age": None,
            "comparison": "",
            "fields": [
                "name",
                "id",
                "email",
                "age",
                "course",
                "year"
            ]
        }

    # =====================================================
    # DETAILS / INFORMATION
    # =====================================================

    if deterministic_name and (
        re.search(
            r"\b(?:tell|show|give)\s+(?:me\s+)?(?:his|her|their)?\s*details\b",
            q
        )
        or re.search(
            r"\b(?:tell|show|give)\s+(?:me\s+)?(?:his|her|their)?\s*information\b",
            q
        )
    ):

        return {
            "intent": "student_fields",
            "student_id": None,
            "student_name": deterministic_name,
            "course": course,
            "year": year,
            "age": None,
            "comparison": "",
            "fields": [
                "name",
                "id",
                "email",
                "age",
                "course",
                "year"
            ]
        }

    # =====================================================
    # STUDENT LOOKUP
    # =====================================================

    if student_name:

        if (
            "is " in q
            and "studying" in q
        ) or (
            "studying" in q
            and "btech" in q
        ):

            return {
                "intent": "verify_course",
                "student_id": None,
                "student_name": student_name,
                "course": course,
                "year": None,
                "age": None,
                "comparison": "",
                "fields": []
            }

        if (
            ("is " in q and " in " in q and "year" in q)
            or ("which year" in q)
            or ("what year" in q)
        ):

            return {
                "intent": "verify_year",
                "student_id": None,
                "student_name": student_name,
                "course": course,
                "year": year,
                "age": None,
                "comparison": "",
                "fields": []
            }

        if fields:

            return {
                "intent": "student_fields",
                "student_id": None,
                "student_name": student_name,
                "course": course,
                "year": year,
                "age": None,
                "comparison": "",
                "fields": fields
            }

    # =====================================================
    # GEMINI
    # =====================================================

    client = get_gemini_client()

    if client:

        prompt = f"""
You are an intent detection system for a Student Database Application.

Analyze the user's question and return ONLY valid JSON.

User question:
{question}

Possible intents:

1. count_students
2. count_by_course
3. count_by_year
4. students_by_course
5. students_by_year
6. student_fields
7. students_fields_list
8. age_comparison
9. oldest_student
10. youngest_student
11. verify_course
12. verify_year
13. verify_email
14. semantic_search
15. unrelated

Allowed student fields:

- name
- id
- email
- age
- course
- year

IMPORTANT RULES:

1. If the question contains:
   "id 1"
   "ID 2"
   "student id 3"
   "student whose ID is 1"
   "student with ID 1"

   the ID is being used to FIND the student.

   Do NOT include "id" in fields unless the user explicitly
   asks for the ID.

2. Only include fields that the user explicitly requested.

3. Never automatically add ID.

4. Example:

   Question:
   "What is the age and email of id 1 student?"

   Return:

   {{
       "intent": "student_fields",
       "student_id": 1,
       "student_name": "",
       "fields": ["age", "email"]
   }}

5. Example:

   Question:
   "Give me the ID of Sahil Dogra"

   Return:

   {{
       "intent": "student_fields",
       "student_id": null,
       "student_name": "Sahil Dogra",
       "fields": ["id"]
   }}

6. Example:

   Question:
   "Give me the name and course of ID 1"

   Return:

   {{
       "intent": "student_fields",
       "student_id": 1,
       "student_name": "",
       "fields": ["name", "course"]
   }}

7. Example:

   Question:
   "Give me all details of ID 1"

   Return:

   {{
       "intent": "student_fields",
       "student_id": 1,
       "student_name": "",
       "fields": [
           "name",
           "id",
           "email",
           "age",
           "course",
           "year"
       ]
   }}

8. If the user explicitly asks for ID, include "id".

9. For age comparison use "age_comparison".

10. For oldest student use "oldest_student".

11. For youngest student use "youngest_student".

12. For general semantic questions about which students
    match a concept or topic, use "semantic_search".

13. Example:

    Question:
    "Who is studying computer science?"

    Return:

    {{
        "intent": "semantic_search",
        "student_id": null,
        "student_name": "",
        "course": "",
        "year": null,
        "age": null,
        "comparison": "",
        "fields": []
    }}

14. Questions such as:

    "Can you find students studying computer science?"
    "Find students studying computer science"
    "List students studying CSE"
    "Tell me everyone studying computer science"
    "I want to find all students who study computer science"

    should use "semantic_search".

15. Questions asking for a year group such as:

    "List all first-year students"
    "Who studies in year 1?"
    "Show me the third-year students"

    should use "students_by_year".

16. Questions asking for BTech CSE or AIML students should use
    "students_by_course".

17. Questions such as:

    "Show me the names of students and the courses they are enrolled in."
    "List all students with their courses."
    "Show me all students and their courses."

    should use:

    "students_fields_list"

18. Questions such as:

    "Does student ID 1 have the email user@gmail.com?"
    "Is user@gmail.com the email of student ID 1?"

    should use:

    "verify_email"

19. For email verification, extract the student ID and expected
    email address.

20. Questions asking "How many [exact course] students are there?"
    should use "count_by_course".

21A. For database matching, treat these two course spellings as equivalent:

    "BTech CSE"

    "BTech - CSE"

    They should be counted and listed together. Do not modify the stored
    database values; normalize only when matching a user's course query.

21. Keep these courses EXACTLY separate:

    "BTech CSE"

    and

    "BTech - CSE"

    They are different course values.

22. Example:

    Question:
    "How many BTech CSE students are there?"

    Return:

    {{
        "intent": "count_by_course",
        "student_id": null,
        "student_name": "",
        "course": "BTech CSE",
        "year": null,
        "age": null,
        "comparison": "",
        "fields": []
    }}

23. Example:

    Question:
    "How many BTech - CSE students are there?"

    Return:

    {{
        "intent": "count_by_course",
        "student_id": null,
        "student_name": "",
        "course": "BTech - CSE",
        "year": null,
        "age": null,
        "comparison": "",
        "fields": []
    }}

Return JSON with these keys:

{{
    "intent": "...",
    "student_id": null,
    "student_name": "",
    "course": "",
    "year": null,
    "age": null,
    "comparison": "",
    "fields": []
}}

Return ONLY JSON.
"""

        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            text = response.text.strip()

            text = re.sub(
                r"```json\s*",
                "",
                text,
                flags=re.IGNORECASE
            )

            text = re.sub(
                r"```\s*$",
                "",
                text
            )

            data = json.loads(
                text
            )

            q = question.lower()

            # =================================================
            # FORCE EXACT COURSE COUNT
            # =================================================

            detected_course_count = (
                extract_course_count_request(
                    question
                )
            )

            if detected_course_count:

                normalized_course = (
                    detected_course_count
                )

                if re.fullmatch(
                    r"btech\s*[-–—]\s*cse",
                    normalized_course,
                    re.IGNORECASE
                ):

                    normalized_course = (
                        "BTech - CSE"
                    )

                elif re.fullmatch(
                    r"btech\s+cse",
                    normalized_course,
                    re.IGNORECASE
                ):

                    normalized_course = (
                        "BTech CSE"
                    )

                elif re.fullmatch(
                    r"aiml",
                    normalized_course,
                    re.IGNORECASE
                ):

                    normalized_course = "AIML"

                elif re.fullmatch(
                    r"btech",
                    normalized_course,
                    re.IGNORECASE
                ):

                    normalized_course = "BTech"

                data["intent"] = (
                    "count_by_course"
                )

                data["student_id"] = None
                data["student_name"] = ""
                data["course"] = normalized_course
                data["year"] = None
                data["age"] = None
                data["comparison"] = ""
                data["fields"] = []

                return data

            # =================================================
            # FORCE NAME + COURSE LIST
            # =================================================

            if is_students_name_course_list_question(
                question
            ):

                data["intent"] = (
                    "students_fields_list"
                )

                data["student_id"] = None
                data["student_name"] = ""
                data["course"] = ""
                data["year"] = None
                data["age"] = None
                data["comparison"] = ""
                data["fields"] = [
                    "name",
                    "course"
                ]

                return data

            # =================================================
            # FORCE EMAIL VERIFICATION
            # =================================================

            detected_student_id = extract_student_id(
                question
            )

            detected_email = extract_email_from_question(
                question
            )

            if (
                detected_student_id is not None
                and detected_email
                and is_email_verification_question(
                    question
                )
            ):

                data["intent"] = (
                    "verify_email"
                )

                data["student_id"] = (
                    detected_student_id
                )

                data["student_name"] = ""

                data["fields"] = []

                return data

            # =================================================
            # FORCE EXPLICIT STUDENT ID
            # =================================================

            if detected_student_id is not None:

                data["intent"] = "student_fields"

                data["student_id"] = (
                    detected_student_id
                )

                data["student_name"] = ""

                local_id_fields = (
                    extract_requested_fields(
                        question
                    )
                )

                if not local_id_fields:

                    local_id_fields = [
                        "name",
                        "id",
                        "email",
                        "age",
                        "course",
                        "year"
                    ]

                data["fields"] = (
                    local_id_fields
                )

                return data

            # =================================================
            # AGE COMPARISON NORMALIZATION
            # =================================================

            age_comparison_match = re.search(
                r"\b(older|younger)\s+than\s+(\d+)\b",
                q
            )

            if age_comparison_match:

                data["intent"] = "age_comparison"

                data["comparison"] = (
                    age_comparison_match.group(1)
                )

                data["age"] = int(
                    age_comparison_match.group(2)
                )

            # =================================================
            # COUNT BY YEAR NORMALIZATION
            # =================================================

            count_year_match = re.search(
                r"\bhow many students\b.*?\b([1-4])(?:st|nd|rd|th)?\s*year\b",
                q
            )

            if count_year_match:

                data["intent"] = "count_by_year"

                data["year"] = int(
                    count_year_match.group(1)
                )

            # =================================================
            # YEAR X NORMALIZATION
            # =================================================

            if data.get("intent") not in {
                "count_by_year",
                "students_by_year"
            }:

                year_number_match = re.search(
                    r"\byear\s+([1-4])\b",
                    q
                )

                if year_number_match:

                    detected_year = int(
                        year_number_match.group(1)
                    )

                    if any(
                        word in q
                        for word in [
                            "who",
                            "which",
                            "students",
                            "student",
                            "list",
                            "show",
                            "tell",
                            "everyone",
                            "find",
                            "study",
                            "studies",
                            "studying",
                            "belong"
                        ]
                    ):

                        data["intent"] = (
                            "students_by_year"
                        )

                        data["year"] = (
                            detected_year
                        )

            # =================================================
            # NATURAL YEAR NORMALIZATION
            # =================================================

            year_patterns = {
                1: [
                    r"\bfirst\s+year\b",
                    r"\bfirst-year\b",
                    r"\b1st\s+year\b"
                ],
                2: [
                    r"\bsecond\s+year\b",
                    r"\bsecond-year\b",
                    r"\b2nd\s+year\b"
                ],
                3: [
                    r"\bthird\s+year\b",
                    r"\bthird-year\b",
                    r"\b3rd\s+year\b"
                ],
                4: [
                    r"\bfourth\s+year\b",
                    r"\bfourth-year\b",
                    r"\b4th\s+year\b"
                ]
            }

            for detected_year, patterns in year_patterns.items():

                if any(
                    re.search(
                        pattern,
                        q
                    )
                    for pattern in patterns
                ):

                    if (
                        "how many" in q
                        or "number of" in q
                        or "count" in q
                    ):

                        data["intent"] = (
                            "count_by_year"
                        )

                    elif any(
                        word in q
                        for word in [
                            "who",
                            "which",
                            "students",
                            "student",
                            "list",
                            "show",
                            "tell",
                            "everyone",
                            "find",
                            "give me",
                            "i want",
                            "study",
                            "studies",
                            "studying",
                            "belong"
                        ]
                    ):

                        data["intent"] = (
                            "students_by_year"
                        )

                    data["year"] = detected_year

                    break

            # =================================================
            # SAFETY CHECK
            # =================================================

            if data.get("intent") == "student_fields":

                fields = data.get(
                    "fields",
                    []
                )

                explicit_id_request = (
                    "what is the id" in q
                    or "what's the id" in q
                    or "give me the id" in q
                    or "show me the id" in q
                    or "tell me the id" in q
                    or "student id" in q
                    or "id of the student" in q
                    or "id of student" in q
                )

                if (
                    "id" in fields
                    and not explicit_id_request
                    and not (
                        "information about" in q
                        or "details about" in q
                        or "all details" in q
                        or "complete details" in q
                        or "full details" in q
                        or "all information" in q
                        or "complete information" in q
                        or "everything about" in q
                    )
                ):

                    fields = [
                        field
                        for field in fields
                        if field != "id"
                    ]

                data["fields"] = list(
                    dict.fromkeys(fields)
                )

            # =================================================
            # FORCE DETERMINISTIC STUDENT NAME
            # =================================================

            deterministic_name = find_student_name_in_question(
                session,
                question
            )

            if deterministic_name:

                data["student_name"] = (
                    deterministic_name
                )

                if any(
                    word in q
                    for word in [
                        "age",
                        "email",
                        "course",
                        "studying",
                        "what course",
                        "what is",
                        "what's",
                        "details",
                        "information"
                    ]
                ):

                    data["intent"] = "student_fields"

                local_fields = extract_requested_fields(
                    question
                )

                if local_fields:

                    data["fields"] = (
                        local_fields
                    )

                if (
                    "studying" in q
                    and (
                        "is " in q
                        or q.startswith("is ")
                    )
                ):

                    data["intent"] = (
                        "verify_course"
                    )

                if (
                    "in 3rd year" in q
                    or "in 2nd year" in q
                    or "in 1st year" in q
                    or "in 4th year" in q
                ):

                    data["intent"] = (
                        "verify_year"
                    )

            return data

        except Exception as e:

            print(
                "Gemini error:",
                e
            )

    # =========================================================
    # LOCAL FALLBACK
    # =========================================================

    q = question.lower()

    student_id = extract_student_id(
        question
    )

    student_name = extract_student_name(
        question
    )

    fields = extract_requested_fields(
        question
    )

    # =========================================================
    # EXACT COURSE COUNT FALLBACK
    # =========================================================

    fallback_course_count = (
        extract_course_count_request(
            question
        )
    )

    if fallback_course_count:

        normalized_course = (
            fallback_course_count
        )

        if re.fullmatch(
            r"btech\s*[-–—]\s*cse",
            normalized_course,
            re.IGNORECASE
        ):

            normalized_course = "BTech - CSE"

        elif re.fullmatch(
            r"btech\s+cse",
            normalized_course,
            re.IGNORECASE
        ):

            normalized_course = "BTech CSE"

        elif re.fullmatch(
            r"aiml",
            normalized_course,
            re.IGNORECASE
        ):

            normalized_course = "AIML"

        elif re.fullmatch(
            r"btech",
            normalized_course,
            re.IGNORECASE
        ):

            normalized_course = "BTech"

        return {
            "intent": "count_by_course",
            "student_id": None,
            "student_name": "",
            "course": normalized_course,
            "year": None,
            "age": None,
            "comparison": "",
            "fields": []
        }

    # =========================================================
    # EMAIL VERIFICATION FALLBACK
    # =========================================================

    fallback_email = extract_email_from_question(
        question
    )

    if (
        student_id is not None
        and fallback_email
        and is_email_verification_question(
            question
        )
    ):

        return {
            "intent": "verify_email",
            "student_id": student_id,
            "student_name": "",
            "course": "",
            "year": None,
            "age": None,
            "comparison": "",
            "fields": []
        }

    # =========================================================
    # ALL STUDENTS NAME + COURSE FALLBACK
    # =========================================================

    if is_students_name_course_list_question(
        question
    ):

        return {
            "intent": "students_fields_list",
            "student_id": None,
            "student_name": "",
            "course": "",
            "year": None,
            "age": None,
            "comparison": "",
            "fields": [
                "name",
                "course"
            ]
        }

    # =========================================================
    # BROAD COMPUTER SCIENCE SEMANTIC FALLBACK
    # =========================================================

    if is_broad_computer_science_question(
        question
    ):

        return {
            "intent": "semantic_search",
            "student_id": None,
            "student_name": "",
            "course": "",
            "year": None,
            "age": None,
            "comparison": "",
            "fields": []
        }

    # =========================================================
    # SEMANTIC SEARCH FALLBACK
    # =========================================================

    semantic_patterns = [

        r"\bwho\s+is\s+studying\s+computer\s+science\b",

        r"\bwho\s+are\s+studying\s+computer\s+science\b",

        r"\bwho\s+studies\s+computer\s+science\b",

        r"\bwhich\s+students\s+study\s+computer\s+science\b",

        r"\bwhich\s+students\s+are\s+studying\s+computer\s+science\b",

        r"\bshow\s+me\s+students\s+studying\s+computer\s+science\b",

        r"\bshow\s+me\s+students\s+studying\s+cse\b",

        r"\bwho\s+studies\s+cse\b",

        r"\bwho\s+is\s+studying\s+cse\b",

        r"\bwhich\s+students\s+study\s+cse\b",

        r"\bwhich\s+students\s+are\s+studying\s+cse\b",

        r"\bcan\s+you\s+find\s+students\s+studying\s+computer\s+science\b",

        r"\bfind\s+students\s+studying\s+computer\s+science\b",

        r"\blist\s+students\s+studying\s+computer\s+science\b",

        r"\btell\s+me\s+which\s+students\s+are\s+studying\s+computer\s+science\b",

        r"\bcan\s+you\s+find\s+students\s+studying\s+cse\b",

        r"\bfind\s+students\s+studying\s+cse\b",

        r"\blist\s+students\s+studying\s+cse\b",

        r"\bwhich\s+students\s+are\s+related\s+to\s+computer\s+science\b",

        r"\bwhich\s+students\s+are\s+related\s+to\s+cse\b"
    ]

    if any(
        re.search(
            pattern,
            q
        )
        for pattern in semantic_patterns
    ):

        return {
            "intent": "semantic_search",
            "student_id": None,
            "student_name": "",
            "course": "",
            "year": None,
            "age": None,
            "comparison": "",
            "fields": []
        }

    # =========================================================
    # SPECIFIC ID MUST NEVER GO TO QDRANT
    # =========================================================

    if student_id is not None:

        if not fields:

            fields = [
                "name",
                "id",
                "email",
                "age",
                "course",
                "year"
            ]

        return {
            "intent": "student_fields",
            "student_id": student_id,
            "student_name": "",
            "course": "",
            "year": None,
            "age": None,
            "comparison": "",
            "fields": fields
        }

    # =========================================================
    # EXACT AGE LOOKUP
    # =========================================================
    # Examples:
    # "Which student is 19 years old?"
    # "Who is 19 years old?"
    # "Which students are 19 years old?"
    # =========================================================

    exact_age_match = re.search(
        r"\b(?:which\s+student|which\s+students|who)\s+(?:is|are)\s+(\d+)\s+years?\s+old\b",
        q
    )

    if exact_age_match:

        return {
            "intent": "age_exact",
            "student_id": student_id,
            "student_name": student_name,
            "course": course,
            "year": year,
            "age": int(exact_age_match.group(1)),
            "comparison": "exact",
            "fields": fields
        }

    # =========================================================
    # OLDEST
    # =========================================================

    if "oldest" in q:

        return {
            "intent": "oldest_student",
            "student_id": student_id,
            "student_name": student_name,
            "fields": fields
        }

    # =========================================================
    # YOUNGEST
    # =========================================================

    if "youngest" in q:

        return {
            "intent": "youngest_student",
            "student_id": student_id,
            "student_name": student_name,
            "fields": fields
        }

    # =========================================================
    # AGE COMPARISON
    # =========================================================

    if (
        "older than" in q
        or "younger than" in q
    ):

        comparison = ""

        if "older than" in q:

            comparison = "older"

        elif "younger than" in q:

            comparison = "younger"

        age_match = re.search(
            r"(?:older than|younger than)\s+(\d+)",
            q
        )

        age = None

        if age_match:

            age = int(
                age_match.group(1)
            )

        return {
            "intent": "age_comparison",
            "student_id": student_id,
            "student_name": student_name,
            "age": age,
            "comparison": comparison,
            "fields": fields
        }

    # =========================================================
    # YEAR
    # =========================================================

    year_match = re.search(
        r"\b([1-4])(?:st|nd|rd|th)?\s*year\b",
        q
    )

    year = None

    if year_match:

        year = int(
            year_match.group(1)
        )

    if year is None:

        year_number_match = re.search(
            r"\byear\s+([1-4])\b",
            q
        )

        if year_number_match:

            year = int(
                year_number_match.group(1)
            )

    # =========================================================
    # COURSE
    # =========================================================

    course = ""

    if "btech" in q:

        if "cse" in q:

            if re.search(
                r"\bbtech\s*[-–—]\s*cse\b",
                q
            ):

                course = "BTech - CSE"

            else:

                course = "BTech CSE"

        else:

            course = "BTech"

    elif "cse" in q:

        course = "CSE"

    # =========================================================
    # AIML COURSE
    # =========================================================

    if re.search(
        r"\baiml\b",
        q
    ):

        course = "AIML"

    # =========================================================
    # NATURAL YEAR QUESTIONS
    # =========================================================

    if year is not None:

        if (
            "how many" in q
            or "number of" in q
            or "count" in q
        ):

            return {
                "intent": "count_by_year",
                "student_id": student_id,
                "student_name": student_name,
                "course": course,
                "year": year,
                "age": None,
                "comparison": "",
                "fields": fields
            }

        if any(
            word in q
            for word in [
                "who",
                "which",
                "students",
                "student",
                "list",
                "show",
                "tell",
                "everyone",
                "find",
                "give me",
                "i want",
                "study",
                "studies",
                "studying",
                "belong"
            ]
        ):

            return {
                "intent": "students_by_year",
                "student_id": student_id,
                "student_name": student_name,
                "course": course,
                "year": year,
                "age": None,
                "comparison": "",
                "fields": fields
            }

    # =========================================================
    # NAME STUDENTS BY COURSE
    # =========================================================
    #
    # Examples:
    # "name all the student in BTech CSE"
    # "name all students in BTech - CSE"
    # "give me the names of students in BTech CSE"
    #
    # This is intentionally handled before Gemini so simple
    # course-list questions cannot be misclassified as a
    # single-student lookup.
    #
    # =========================================================

    if course and re.search(
        r"\bname(?:s)?\b",
        q
    ) and re.search(
        r"\bstudents?\b",
        q
    ):

        return {
            "intent": "students_by_course",
            "student_id": student_id,
            "student_name": "",
            "course": course,
            "year": None,
            "age": None,
            "comparison": "",
            "fields": []
        }

    # =========================================================
    # NATURAL COURSE QUESTIONS
    # =========================================================

    if course:

        if (
            "how many" in q
            or "number of" in q
            or "count" in q
        ):

            return {
                "intent": "count_by_course",
                "student_id": student_id,
                "student_name": student_name,
                "course": course,
                "year": year,
                "age": None,
                "comparison": "",
                "fields": fields
            }

        if any(
            phrase in q
            for phrase in [
                "which students",
                "which student",
                "students studying",
                "students are studying",
                "everyone",
                "every student",
                "every students",
                "every btech",
                "who all",
                "all students",
                "any students",
                "are there any",
                "find students",
                "list students",
                "tell me everyone",
                "i want a list"
            ]
        ):

            return {
                "intent": "students_by_course",
                "student_id": student_id,
                "student_name": student_name,
                "course": course,
                "year": year,
                "age": None,
                "comparison": "",
                "fields": fields
            }

    # =========================================================
    # STUDENT FIELDS
    # =========================================================

    if fields:

        return {
            "intent": "student_fields",
            "student_id": student_id,
            "student_name": student_name,
            "course": course,
            "year": year,
            "age": None,
            "comparison": "",
            "fields": fields
        }

    # =========================================================
    # VERIFY YEAR
    # =========================================================

    if (
        "which year" in q
        or "what year" in q
        or (
            "is in" in q
            and "year" in q
        )
    ):

        return {
            "intent": "verify_year",
            "student_id": student_id,
            "student_name": student_name,
            "course": course,
            "year": year,
            "age": None,
            "comparison": "",
            "fields": fields
        }

    # =========================================================
    # COUNT STUDENTS
    # =========================================================

    if (
        "how many students" in q
        or "total students" in q
        or "number of students" in q
    ):

        return {
            "intent": "count_students",
            "student_id": student_id,
            "student_name": student_name,
            "course": course,
            "year": year,
            "age": None,
            "comparison": "",
            "fields": fields
        }

    # =========================================================
    # STUDENTS BY COURSE
    # =========================================================

    if (
        "students in" in q
        or "who is studying" in q
        or (
            "which students" in q
            and course
        )
    ):

        return {
            "intent": "students_by_course",
            "student_id": student_id,
            "student_name": student_name,
            "course": course,
            "year": year,
            "age": None,
            "comparison": "",
            "fields": fields
        }

    # =========================================================
    # VERIFY COURSE
    # =========================================================

    if (
        "which course" in q
        or "what course" in q
        or "course is" in q
    ):

        return {
            "intent": "verify_course",
            "student_id": student_id,
            "student_name": student_name,
            "course": course,
            "year": year,
            "age": None,
            "comparison": "",
            "fields": fields
        }

    # =========================================================
    # UNRELATED
    # =========================================================

    unrelated_words = [
        "weather",
        "joke",
        "movie",
        "football",
        "cricket",
        "news"
    ]

    if any(
        word in q
        for word in unrelated_words
    ):

        return {
            "intent": "unrelated",
            "student_id": student_id,
            "student_name": student_name,
            "course": course,
            "year": year,
            "age": None,
            "comparison": "",
            "fields": fields
        }

    # =========================================================
    # UNKNOWN
    # =========================================================

    return {
        "intent": "unknown",
        "student_id": student_id,
        "student_name": student_name,
        "course": course,
        "year": year,
        "age": None,
        "comparison": "",
        "fields": fields
    }


# =========================================================
# FIND STUDENT
# =========================================================

def find_student(
    session: Session,
    student_id=None,
    student_name=""
):

    # =====================================================
    # SEARCH BY ID
    # =====================================================

    if student_id is not None:

        student = session.get(
            Student,
            student_id
        )

        if student:

            return student

    # =====================================================
    # SEARCH BY NAME
    # =====================================================

    if student_name:

        statement = select(
            Student
        ).where(
            Student.name.ilike(
                student_name
            )
        )

        student = session.exec(
            statement
        ).first()

        if student:

            return student

        statement = select(
            Student
        ).where(
            Student.name.ilike(
                f"%{student_name}%"
            )
        )

        student = session.exec(
            statement
        ).first()

        if student:

            return student

    return None


# =========================================================
# ORDINAL YEAR
# =========================================================

def ordinal_year(year):

    if year == 1:
        return "1st year"

    if year == 2:
        return "2nd year"

    if year == 3:
        return "3rd year"

    if year == 4:
        return "4th year"

    return f"{year}th year"


# =========================================================
# SELECTED STUDENT FIELDS
# =========================================================

def selected_student_fields(
    student: Student,
    fields: list[str]
):

    output = []

    fields = list(
        dict.fromkeys(fields)
    )

    for field in fields:

        if field == "name":

            output.append(
                f"Name: {student.name}"
            )

        elif field == "id":

            output.append(
                f"ID: {student.id}"
            )

        elif field == "email":

            output.append(
                f"Email: {student.email}"
            )

        elif field == "age":

            output.append(
                f"Age: {student.age}"
            )

        elif field == "course":

            output.append(
                f"Course: {student.course}"
            )

        elif field == "year":

            output.append(
                f"Year: {student.year}"
            )

    return "\n".join(
        output
    )


# =========================================================
# PROCESS QUESTION
# =========================================================

def process_question(
    state: ChatState
):

    intent = state["intent"]

    student_id = state["student_id"]

    student_name = state["student_name"]

    course = state["course"]

    year = state["year"]

    age = state["age"]

    comparison = state["comparison"]

    fields = state["fields"]

    session = state["session"]

    # =====================================================
    # ALL STUDENTS NAME + COURSE
    # =====================================================

    if intent == "students_fields_list":

        students = session.exec(
            select(Student)
        ).all()

        if not students:

            return {
                "answer":
                    "No students found in the database."
            }

        lines = []

        for student in students:

            lines.append(
                f"- {student.name} — {student.course}"
            )

        return {
            "answer":
                "Students and their courses:\n"
                + "\n".join(lines)
        }

    # =====================================================
    # VERIFY EMAIL
    # =====================================================

    if intent == "verify_email":

        student = find_student(
            session,
            student_id,
            student_name
        )

        if not student:

            return {
                "answer":
                    "Student not found."
            }

        expected_email = extract_email_from_question(
            state["question"]
        )

        if not expected_email:

            return {
                "answer":
                    "Please provide an email address to verify."
            }

        actual_email = str(
            student.email
        ).strip()

        if (
            actual_email.casefold()
            == expected_email.strip().casefold()
        ):

            return {
                "answer":
                    f"Yes, student ID {student.id} has the email {actual_email}."
            }

        return {
            "answer":
                f"No, student ID {student.id} has the email {actual_email}, not {expected_email}."
        }

    # =====================================================
    # STUDENT FIELDS
    # =====================================================

    if intent == "student_fields":

        student = find_student(
            session,
            student_id,
            student_name
        )

        if not student:

            return {
                "answer":
                    "Student not found."
            }

        if not fields:

            fields = [
                "name",
                "id",
                "email",
                "age",
                "course",
                "year"
            ]

        return {
            "answer":
                selected_student_fields(
                    student,
                    fields
                )
        }

    # =====================================================
    # COUNT BY YEAR
    # =====================================================

    if intent == "count_by_year":

        if year is None:

            return {
                "answer":
                    "Please specify a year."
            }

        students = session.exec(
            select(Student)
        ).all()

        matching = [
            student
            for student in students
            if student.year == year
        ]

        count = len(matching)

        if count == 1:

            return {
                "answer":
                    f"There is 1 student in {ordinal_year(year)}."
            }

        return {
            "answer":
                f"There are {count} students in {ordinal_year(year)}."
        }

    # =====================================================
    # COUNT STUDENTS
    # =====================================================

    if intent == "count_students":

        students = session.exec(
            select(Student)
        ).all()

        count = len(students)

        if count == 1:

            return {
                "answer":
                    "There is 1 student in the database."
            }

        return {
            "answer":
                f"There are {count} students in the database."
        }

    # =====================================================
    # COUNT BY COURSE
    # =====================================================

    if intent == "count_by_course":

        if not course:

            return {
                "answer":
                    "Please specify a course."
            }

        students = session.exec(
            select(Student)
        ).all()

        matching = [
            student
            for student in students
            if course_matches_requested(
                student.course,
                course
            )
        ]

        count = len(matching)

        # =================================================
        # FIXED SINGULAR / PLURAL GRAMMAR
        # =================================================

        if count == 1:

            return {
                "answer":
                    f"There is 1 student in {course}."
            }

        return {
            "answer":
                f"There are {count} students in {course}."
        }

    # =====================================================
    # STUDENTS BY COURSE
    # =====================================================

    if intent == "students_by_course":

        if not course:

            return {
                "answer":
                    "Please specify a course."
            }

        all_students = session.exec(
            select(Student)
        ).all()

        students = [
            student
            for student in all_students
            if course_matches_requested(
                student.course,
                course
            )
        ]

        if not students:

            return {
                "answer":
                    f"No students found in {course}."
            }

        names = [
            student.name
            for student in students
        ]

        return {
            "answer":
                f"Students in {course}:\n"
                + "\n".join(
                    f"- {name}"
                    for name in names
                )
        }

    # =====================================================
    # STUDENTS BY YEAR
    # =====================================================

    if intent == "students_by_year":

        if year is None:

            return {
                "answer":
                    "Please specify a year."
            }

        statement = select(
            Student
        ).where(
            Student.year == year
        )

        students = session.exec(
            statement
        ).all()

        if not students:

            return {
                "answer":
                    f"No students found in {ordinal_year(year)}."
            }

        names = [
            student.name
            for student in students
        ]

        return {
            "answer":
                f"Students in {ordinal_year(year)}:\n"
                + "\n".join(
                    f"- {name}"
                    for name in names
                )
        }

    # =====================================================
    # EXACT AGE LOOKUP
    # =====================================================

    if intent == "age_exact":

        if age is None:

            return {
                "answer":
                    "Please specify an age."
            }

        students = session.exec(
            select(Student)
        ).all()

        matching = [
            student
            for student in students
            if student.age == age
        ]

        if not matching:

            return {
                "answer":
                    f"No students are {age} years old."
            }

        lines = [
            f"- {student.name} ({student.age})"
            for student in matching
        ]

        return {
            "answer":
                f"Students who are {age} years old:\n"
                + "\n".join(lines)
        }

    # =====================================================
    # AGE COMPARISON
    # =====================================================

    if intent == "age_comparison":

        if age is None:

            return {
                "answer":
                    "Please specify an age."
            }

        students = session.exec(
            select(Student)
        ).all()

        if comparison == "older":

            matching = [
                student
                for student in students
                if student.age > age
            ]

            condition = (
                f"older than {age}"
            )

        elif comparison == "younger":

            matching = [
                student
                for student in students
                if student.age < age
            ]

            condition = (
                f"younger than {age}"
            )

        else:

            return {
                "answer":
                    "Please specify whether you mean older or younger."
            }

        if not matching:

            return {
                "answer":
                    f"No students are {condition}."
            }

        lines = [
            f"- {student.name} ({student.age})"
            for student in matching
        ]

        return {
            "answer":
                f"Students {condition}:\n"
                + "\n".join(lines)
        }

    # =====================================================
    # OLDEST STUDENT
    # =====================================================

    if intent == "oldest_student":

        students = session.exec(
            select(Student)
        ).all()

        if not students:

            return {
                "answer":
                    "No students found."
            }

        oldest = max(
            students,
            key=lambda student: student.age
        )

        return {
            "answer":
                f"The oldest student is "
                f"{oldest.name}, aged {oldest.age}."
        }

    # =====================================================
    # YOUNGEST STUDENT
    # =====================================================

    if intent == "youngest_student":

        students = session.exec(
            select(Student)
        ).all()

        if not students:

            return {
                "answer":
                    "No students found."
            }

        youngest = min(
            students,
            key=lambda student: student.age
        )

        return {
            "answer":
                f"The youngest student is "
                f"{youngest.name}, aged {youngest.age}."
        }

    # =====================================================
    # VERIFY COURSE
    # =====================================================

    if intent == "verify_course":

        student = find_student(
            session,
            student_id,
            student_name
        )

        if not student:

            return {
                "answer":
                    "Student not found."
            }

        q = state["question"].strip().casefold()

        if q.startswith(
            ("is ", "does ")
        ) and course:

            matches = (
                student.course.strip().casefold()
                == course.strip().casefold()
            )

            # Treat BTech CSE and BTech - CSE as equivalent here too.
            matches = course_matches_requested(
                student.course,
                course
            )

            if matches:

                return {
                    "answer":
                        f"Yes, {student.name} is studying {student.course}."
                }

            return {
                "answer":
                    f"No, {student.name} is studying {student.course}, not {course}."
            }

        return {
            "answer":
                f"{student.name} is studying {student.course}."
        }

    # =====================================================
    # VERIFY YEAR
    # =====================================================

    if intent == "verify_year":

        student = find_student(
            session,
            student_id,
            student_name
        )

        if not student:

            return {
                "answer":
                    "Student not found."
            }

        q = state["question"].strip().casefold()

        if q.startswith(
            ("is ", "does ")
        ) and year is not None:

            if student.year == year:

                return {
                    "answer":
                        f"Yes, {student.name} is in {ordinal_year(student.year)}."
                }

            return {
                "answer":
                    f"No, {student.name} is in {ordinal_year(student.year)}, not {ordinal_year(year)}."
            }

        return {
            "answer":
                f"{student.name} is in "
                f"{ordinal_year(student.year)}."
        }

    # =====================================================
    # SEMANTIC SEARCH
    # =====================================================

    if intent == "semantic_search":

        vector_results = state.get(
            "vector_results",
            []
        )

        if not vector_results:

            return {
                "answer":
                    "I could not find relevant students."
            }

        lines = []

        question = state["question"].casefold()

        show_names_and_years = (
            "name" in question
            and "year" in question
        )

        for student in vector_results:

            name = student.get("name")

            course = student.get("course")

            year_value = student.get("year")

            if show_names_and_years:

                if (
                    name
                    and year_value is not None
                ):

                    lines.append(
                        f"- {name} — {ordinal_year(year_value)}"
                    )

                elif name:

                    lines.append(
                        f"- {name}"
                    )

            else:

                if (
                    name
                    and course
                    and year_value is not None
                ):

                    lines.append(
                        f"- {name} ({course}, {ordinal_year(year_value)})"
                    )

                elif name and course:

                    lines.append(
                        f"- {name} ({course})"
                    )

                elif name:

                    lines.append(
                        f"- {name}"
                    )

        if not lines:

            return {
                "answer":
                    "I found student records, but could not format the result."
            }

        return {
            "answer":
                "Students matching your question:\n"
                + "\n".join(lines)
        }

    # =====================================================
    # UNSUPPORTED STUDENT FIELD
    # =====================================================

    if intent == "unsupported_student_field":

        unsupported_field = state.get(
            "unsupported_field",
            "requested information"
        )

        return {
            "answer":
                f"The student database does not contain {unsupported_field} information. "
                "Available fields are name, ID, email, age, course, and year."
        }

    # =====================================================
    # UNRELATED
    # =====================================================

    if intent == "unrelated":

        return {
            "answer":
                "I can only answer questions related to the student database."
        }

    # =====================================================
    # UNKNOWN
    # =====================================================

    return {
        "answer":
            "I could not understand the question. "
            "Please ask something related to the student database."
    }


# =========================================================
# LANGGRAPH WORKFLOW
# =========================================================

def understand_node(
    state: ChatState
):

    intent_data = understand_question(
        state["question"],
        state["session"]
    )

    state.update({

        "intent": intent_data.get(
            "intent",
            ""
        ),

        "student_name": intent_data.get(
            "student_name",
            ""
        ),

        "student_id": intent_data.get(
            "student_id"
        ),

        "course": intent_data.get(
            "course",
            ""
        ),

        "year": intent_data.get(
            "year"
        ),

        "age": intent_data.get(
            "age"
        ),

        "comparison": intent_data.get(
            "comparison",
            ""
        ),

        "fields": intent_data.get(
            "fields",
            []
        )

        ,
        "unsupported_field": intent_data.get(
            "unsupported_field",
            ""
        )
    })

    # =====================================================
    # DATABASE NAME HAS PRIORITY
    # =====================================================

    db_student_name = find_student_name_in_question(
        state["session"],
        state["question"]
    )

    if db_student_name:

        state["student_name"] = (
            db_student_name
        )

        named_fields = extract_requested_fields(
            state["question"]
        )

        if (
            named_fields
            and state["intent"] in {
                "unknown",
                "unrelated",
                "count_students"
            }
        ):

            state["intent"] = (
                "student_fields"
            )

            state["fields"] = (
                named_fields
            )

    # =====================================================
    # COURSE COUNT HAS PRIORITY
    # =====================================================

    detected_course_count = (
        extract_course_count_request(
            state["question"]
        )
    )

    if detected_course_count:

        normalized_course = (
            detected_course_count
        )

        if re.fullmatch(
            r"btech\s*[-–—]\s*cse",
            normalized_course,
            re.IGNORECASE
        ):

            normalized_course = "BTech - CSE"

        elif re.fullmatch(
            r"btech\s+cse",
            normalized_course,
            re.IGNORECASE
        ):

            normalized_course = "BTech CSE"

        elif re.fullmatch(
            r"aiml",
            normalized_course,
            re.IGNORECASE
        ):

            normalized_course = "AIML"

        elif re.fullmatch(
            r"btech",
            normalized_course,
            re.IGNORECASE
        ):

            normalized_course = "BTech"

        state["intent"] = "count_by_course"
        state["student_id"] = None
        state["student_name"] = ""
        state["course"] = normalized_course
        state["year"] = None
        state["age"] = None
        state["comparison"] = ""
        state["fields"] = []

    # =====================================================
    # DATABASE ID HAS HIGHEST PRIORITY
    # =====================================================

    db_student_id = extract_student_id(
        state["question"]
    )

    if db_student_id is not None:

        state["student_id"] = (
            db_student_id
        )

        state["student_name"] = ""

        # =================================================
        # FULL DETAILS REQUEST WITH STUDENT ID
        # =================================================
        # Phrases such as "Tell me the details of student ID 1"
        # should return the complete record, not only the ID.
        # Keep the existing ID handling below unchanged.
        id_details_question = state["question"].strip().casefold()

        if (
            "details of" in id_details_question
            or "details for" in id_details_question
            or "information of" in id_details_question
            or "information for" in id_details_question
        ):

            state["intent"] = "student_fields"
            state["fields"] = [
                "name",
                "id",
                "email",
                "age",
                "course",
                "year"
            ]

        # =================================================
        # DO NOT OVERRIDE EMAIL VERIFICATION
        # =================================================

        if (
            state["intent"] == "verify_email"
            and extract_email_from_question(
                state["question"]
            )
        ):

            state["fields"] = []

        # Course count should also not be overridden.
        elif state["intent"] == "count_by_course":

            pass

        # Full details requests should keep all fields.
        elif (
            state["intent"] == "student_fields"
            and state["fields"] == [
                "name",
                "id",
                "email",
                "age",
                "course",
                "year"
            ]
            and (
                "details of" in state["question"].strip().casefold()
                or "details for" in state["question"].strip().casefold()
                or "information of" in state["question"].strip().casefold()
                or "information for" in state["question"].strip().casefold()
            )
        ):

            pass

        # Unsupported fields must never fall through to the generic
        # student_fields/ID lookup. Keep the intent selected by the
        # early unsupported-field detection above.
        elif state["intent"] == "unsupported_student_field":

            pass

        else:

            state["intent"] = (
                "student_fields"
            )

            id_fields = extract_requested_fields(
                state["question"]
            )

            # Keep ID-only questions as full student-detail lookups.
            # This prevents "Student ID 1" from being reduced to
            # only the ID field after the LangGraph state update.
            q_clean = state["question"].strip().casefold()

            id_only_patterns = [
                r"^student\s+id\s*(?:is|=|:|#)?\s*\d+\s*[?.!]*$",
                r"^id\s*(?:is|=|:|#)?\s*\d+\s*[?.!]*$",
                r"^student\s+whose\s+id\s*(?:is|=|:|#)?\s*\d+\s*[?.!]*$",
                r"^student\s+with\s+id\s*(?:is|=|:|#)?\s*\d+\s*[?.!]*$"
            ]

            is_id_only_question = any(
                re.fullmatch(pattern, q_clean)
                for pattern in id_only_patterns
            )

            if is_id_only_question:
                id_fields = [
                    "name",
                    "id",
                    "email",
                    "age",
                    "course",
                    "year"
                ]

            if not id_fields:

                id_fields = [
                    "name",
                    "id",
                    "email",
                    "age",
                    "course",
                    "year"
                ]

            state["fields"] = id_fields

    return state


# =========================================================
# QDRANT SEMANTIC SEARCH NODE
# =========================================================

def semantic_search_node(
    state: ChatState
):

    if state["intent"] not in {
        "unknown",
        "semantic_search"
    }:

        return state

    question = (
        state["question"]
        .strip()
        .casefold()
    )

    # =====================================================
    # NEVER SEND STUDENT-ID QUESTIONS TO QDRANT
    # =====================================================

    if extract_student_id(
        state["question"]
    ) is not None:

        return state

    # =====================================================
    # DO NOT SEND STRUCTURED QUESTIONS TO QDRANT
    # =====================================================

    if re.search(
        r"\b(?:first|1st|second|2nd|third|3rd|fourth|4th)[-\s]+year\b",
        question
    ):

        return state

    if re.search(
        r"\byear\s+(?:one|1|two|2|three|3|four|4)\b",
        question
    ):

        return state

    # =====================================================
    # COURSE QUESTIONS
    # =====================================================

    if re.search(
        r"\baiml\b",
        question
    ):

        return state

    if re.search(
        r"\bbtech\s*[-–—]?\s*cse\b",
        question
    ):

        return state

    # =====================================================
    # STUDENT KEYWORDS
    # =====================================================

    student_keywords = [

        "student",
        "students",
        "study",
        "studies",
        "studying",
        "course",
        "class",
        "college",
        "computer science",
        "cse",
        "aiml"
    ]

    is_student_question = any(
        keyword in question
        for keyword in student_keywords
    )

    if not is_student_question:

        return state

    # =====================================================
    # QDRANT SEARCH
    # =====================================================

    try:

        results = search_students(
            state["question"],
            limit=3
        )

        state["vector_results"] = []

        for result in results:

            payload = (
                result.payload
                or {}
            )

            state["vector_results"].append({

                "score": getattr(
                    result,
                    "score",
                    None
                ),

                "student_id": payload.get(
                    "student_id"
                ),

                "name": payload.get(
                    "name"
                ),

                "email": payload.get(
                    "email"
                ),

                "age": payload.get(
                    "age"
                ),

                "course": payload.get(
                    "course"
                ),

                "year": payload.get(
                    "year"
                )
            })

        if state["vector_results"]:

            state["intent"] = (
                "semantic_search"
            )

    except Exception as e:

        print(
            "Qdrant search error:",
            e
        )

        state["vector_results"] = []

    return state


# =========================================================
# PROCESS NODE
# =========================================================

def process_node(
    state: ChatState
):

    result = process_question(
        state
    )

    state["answer"] = (
        result["answer"]
    )

    return state


# =========================================================
# LANGGRAPH GRAPH
# =========================================================

chatbot_workflow = StateGraph(
    ChatState
)

chatbot_workflow.add_node(
    "understand_question",
    understand_node
)

chatbot_workflow.add_node(
    "semantic_search",
    semantic_search_node
)

chatbot_workflow.add_node(
    "process_question",
    process_node
)

chatbot_workflow.add_edge(
    START,
    "understand_question"
)

chatbot_workflow.add_edge(
    "understand_question",
    "semantic_search"
)

chatbot_workflow.add_edge(
    "semantic_search",
    "process_question"
)

chatbot_workflow.add_edge(
    "process_question",
    END
)

chatbot_graph = (
    chatbot_workflow.compile()
)


# =========================================================
# MAIN ANSWER FUNCTION
# =========================================================

def answer_question(
    question: str,
    session: Session
):

    state: ChatState = {

        "question": question,

        "intent": "",

        "student_name": "",

        "student_id": None,

        "course": "",

        "year": None,

        "age": None,

        "comparison": "",

        "fields": [],

        "answer": "",

        "session": session,

        "vector_results": []
    }

    result_state = chatbot_graph.invoke(
        state
    )

    return result_state[
        "answer"
    ]


# =========================================================
# API ENDPOINT
# =========================================================

@router.post("/ask")
def ask_chatbot(
    request: ChatRequest,
    session: Session = Depends(
        get_session
    )
):

    answer = answer_question(
        request.question,
        session
    )

    return {
        "answer": answer
    }
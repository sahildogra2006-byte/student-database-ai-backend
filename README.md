# Student Database Application System

A modular Student Database Application built with **Python, FastAPI, SQLite, LangGraph, Gemini, and Qdrant**. The project provides REST APIs for student management and an AI-powered chatbot that can answer natural-language questions about student records.

---

## Project Overview

This project was developed as an AIML internship project and combines:

* FastAPI backend development
* Student CRUD operations
* SQLite database
* Request validation
* Swagger/OpenAPI documentation
* AI chatbot integration
* LangGraph workflow
* Gemini integration
* Qdrant vector database and semantic search
* Simple web frontend
* Testing and documentation

The application allows users to manage student records and ask questions about the stored data using natural language.

---

## Features

### Student Management

The backend supports complete CRUD operations:

* Create a student
* View all students
* View a student by ID
* Update student information
* Delete a student

Student fields include:

* ID
* Name
* Email
* Age
* Course
* Year

### AI Student Assistant

The chatbot accepts natural-language questions about the student database.

Examples:

```text
Which students are in 3rd year?
What is the email of ID 1?
How old is the student with ID 1?
Which course is Student A studying?
How many BTech CSE students are there?
Show all students.
Who is the oldest student?
```

The chatbot supports both individual student questions and database-level questions.

### Course Matching

The chatbot treats equivalent course spellings such as:

```text
BTech CSE
BTech - CSE
```

as the same course for relevant database queries.

### Semantic Search

Qdrant is used for semantic search over student information. This allows natural-language queries to retrieve relevant student records based on meaning rather than only exact keyword matching.

Vector search is used where it is useful, while structured database operations continue to handle questions that are better answered directly from SQLite.

---

## Technology Stack

| Technology          | Purpose                             |
| ------------------- | ----------------------------------- |
| Python              | Main programming language           |
| FastAPI             | Backend REST API                    |
| SQLModel            | Database models and ORM             |
| SQLite              | Student database                    |
| Pydantic            | Request/data validation             |
| Gemini              | Natural-language AI responses       |
| LangGraph           | Chatbot workflow orchestration      |
| Qdrant              | Vector database and semantic search |
| FastEmbed           | Local text embedding                |
| HTML/CSS/JavaScript | Frontend                            |
| Uvicorn             | ASGI development server             |
| Swagger/OpenAPI     | API documentation                   |

---

## Project Structure

```text
student_database_backend_starter/
├── ai/
│   ├── __init__.py
│   ├── chatbot.py
│   └── vector_store.py
├── routers/
│   ├── students.py
│   └── chatbot.py
├── frontend/
│   └── index.html
├── main.py
├── database.py
├── models.py
├── schemas.py
├── crud.py
├── students.db
├── requirements.txt
├── test_gemini.py
└── README.md
```

---

## Backend Architecture

The application is organized into separate modules.

### `main.py`

Creates the FastAPI application and registers the application routers.

### `database.py`

Creates the SQLite database engine and provides database sessions.

### `models.py`

Defines the SQLModel student database model.

### `schemas.py`

Defines request and response data structures.

### `crud.py`

Contains database CRUD operations.

### `routers/students.py`

Provides REST API endpoints for student management.

### `routers/chatbot.py`

Provides the chatbot API endpoint.

### `ai/chatbot.py`

Contains the chatbot and AI-related logic.

### `ai/vector_store.py`

Handles Qdrant collection creation, student synchronization, embeddings, and semantic search.

### `frontend/index.html`

Provides the browser-based interface for viewing students and interacting with the AI Student Assistant.

---

## Chatbot Workflow

The chatbot uses a workflow that connects the user's question with the appropriate student-data retrieval process.

```text
User Question
      |
      v
LangGraph Workflow
      |
      v
Determine Question / Retrieval Operation
      |
      +----------------------+
      |                      |
      v                      v
Structured Database      Semantic Search
(SQLite)                 (Qdrant)
      |                      |
      +----------+-----------+
                 |
                 v
        Student Information
                 |
                 v
          Gemini / Response
                 |
                 v
       Natural-Language Answer
```

For structured questions such as student ID, age, email, course, year, and counts, the application can use direct database logic.

For semantic questions, Qdrant can retrieve relevant student records.

---

## Database

The project uses SQLite through SQLModel.

Default database:

```text
students.db
```

Student model:

```text
ID
Name
Email
Age
Course
Year
```

The database is automatically initialized when the FastAPI application starts.

---

## API Endpoints

### Root

```http
GET /
```

Returns a message confirming that the backend is running.

### Health Check

```http
GET /health
```

Returns:

```json
{
  "status": "ok"
}
```

### Get All Students

```http
GET /students/
```

Returns all student records.

### Get Student by ID

```http
GET /students/{student_id}
```

Returns a specific student using the student ID.

### Create Student

```http
POST /students/
```

Example request:

```json
{
  "name": "Student A",
  "email": "student@example.com",
  "age": 20,
  "course": "BTech CSE",
  "year": 3
}
```

### Update Student

```http
PUT /students/{student_id}
```

Updates the information of an existing student.

### Delete Student

```http
DELETE /students/{student_id}
```

Deletes a student from the database.

### AI Chatbot

```http
POST /chatbot/ask
```

Example request:

```json
{
  "question": "Which students are in 3rd year?"
}
```

---

## Swagger / OpenAPI

FastAPI automatically provides interactive API documentation.

After starting the backend, open:

```text
http://127.0.0.1:8000/docs
```

Swagger can be used to test:

* Student CRUD APIs
* Health endpoint
* Chatbot endpoint

The OpenAPI schema is also available at:

```text
http://127.0.0.1:8000/openapi.json
```

---

## Running the Project

### 1. Open the project folder

```powershell
cd "C:\Users\Dell\Downloads\student_database_backend_starter"
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

For PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file if AI configuration is required.

Example:

```text
GEMINI_API_KEY=your_api_key_here
```

Do not commit the actual API key to GitHub.

### 6. Start the backend

```powershell
uvicorn main:app --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

### 7. Open Swagger

```text
http://127.0.0.1:8000/docs
```

### 8. Open the frontend

Open:

```text
frontend/index.html
```

in a browser while the FastAPI server is running.

---

## Qdrant Vector Database

The project uses **Qdrant** as the vector database for semantic search.

### Why Qdrant?

Qdrant was selected because it provides:

* Easy Python integration
* Semantic similarity search
* FastEmbed compatibility
* Local development support
* Simple collection management
* Good compatibility with AI applications
* Local storage without requiring a separately hosted database during development

### Qdrant Configuration

Collection name:

```text
student_information
```

Embedding model:

```text
BAAI/bge-small-en-v1.5
```

Local storage:

```text
qdrant_data/
```

### Student Synchronization

Student records from SQLite can be synchronized into Qdrant.

The vector store converts student information into searchable text containing:

```text
Student ID
Name
Email
Age
Course
Year
```

This information is converted into embeddings and stored in the Qdrant collection.

### Semantic Search Example

A query such as:

```text
Which students study computer science?
```

can retrieve relevant student records based on semantic similarity.

### Why Vector Search Is Not Used for Every Query

Not every database question requires semantic search.

For example:

```text
What is the email of ID 1?
```

is better handled directly using the structured SQLite database.

A semantic question such as:

```text
Which students study computer science?
```

can benefit from vector search.

This combination allows the project to use structured database queries where appropriate and semantic retrieval where useful.

---

## Gemini Integration

Gemini is used for natural-language processing and response generation where applicable.

The chatbot is designed to:

1. Receive the user's question.
2. Determine the relevant student-data operation.
3. Retrieve information from the database or semantic search system.
4. Generate a natural-language response.
5. Return the answer through the FastAPI chatbot endpoint.

The application uses an environment variable for the Gemini API key.

Example:

```text
GEMINI_API_KEY=your_api_key_here
```

Never publish an actual API key in the GitHub repository.

---

## LangGraph Integration

LangGraph is used to organize the chatbot workflow.

The workflow connects the user's question with the appropriate student-data retrieval process.

A simplified flow is:

```text
User Prompt
    |
    v
Question / Intent Processing
    |
    v
Database or Retrieval Operation
    |
    v
Student Data
    |
    v
Response Generation
    |
    v
Final Answer
```

This provides a modular workflow instead of placing all chatbot logic inside a single function.

---

## Frontend

The project includes a simple browser-based frontend.

The frontend provides:

* Student database table
* Refresh button
* AI Student Assistant
* Question input
* Chatbot response display

The frontend communicates with the FastAPI backend through HTTP requests.

---

## Testing

The project was tested using FastAPI Swagger/OpenAPI and direct API requests.

Testing included:

### Student ID and Details

Examples:

```text
What is the ID of Student A?
What is the age and email of ID 1?
```

### Name Queries

Examples:

```text
Tell me about Student A.
Which student has this email?
```

### Email Verification

Examples:

```text
What is the email of ID 1?
Does Student A have student@example.com?
```

### Age Queries

Examples:

```text
Which student is 19 years old?
Who is the oldest student?
```

### Course Queries

Examples:

```text
Which course is Student A studying?
Which students study BTech CSE?
```

### Year Queries

Examples:

```text
Which students are in 3rd year?
Which students are in 1st year?
```

### Count Queries

Examples:

```text
How many BTech students are there?
How many BTech CSE students are there?
```

### All Student Queries

Examples:

```text
Show all students.
List all students.
```

### Course Matching

The chatbot was tested with equivalent course formats such as:

```text
BTech CSE
BTech - CSE
```

These are treated as equivalent for relevant queries.

### Semantic Search

Qdrant semantic search was tested with natural-language questions about student information.

### Unsupported Fields

Questions about fields that are not part of the student model, such as salary or phone number, are not treated as valid student fields.

### Invalid IDs

Non-existent student IDs are handled without inventing student information.

---

## Requirements

Install the project dependencies with:

```powershell
pip install -r requirements.txt
```

The requirements include the libraries needed for:

* FastAPI
* Uvicorn
* SQLModel
* Pydantic
* Gemini
* LangGraph
* LangChain Google GenAI
* Qdrant
* FastEmbed

---

## Environment Variables

Use a `.env` file for sensitive configuration.

Example:

```text
GEMINI_API_KEY=your_api_key_here
```

The `.env` file should not be committed to GitHub.

---

## Security

Before uploading the project to GitHub:

* Do not commit `.env`
* Do not commit API keys
* Do not commit passwords
* Do not commit private credentials
* Do not expose production secrets in source code
* Use `.gitignore` for local-only files

Recommended entries in `.gitignore` include:

```text
.venv/
__pycache__/
.env
qdrant_data/
```

If `students.db` contains private or real personal information, it should also be excluded from the public repository or replaced with safe sample data.

---

## Deployment

The backend can be deployed as a FastAPI application using an ASGI server such as Uvicorn.

For production deployment, environment variables should be configured securely and API credentials should never be stored directly in source code.

The local development command is:

```powershell
uvicorn main:app --reload
```

---

## Future Improvements

Possible future improvements include:

* Authentication and authorization
* Better frontend styling
* More advanced student analytics
* Additional database filters
* Improved chatbot intent classification
* More advanced LangGraph agents
* Hosted Qdrant deployment
* Production database such as PostgreSQL
* Automated unit and integration tests
* Deployment to a cloud platform

---

## Project Goal

The goal of this project is to demonstrate how a modular FastAPI backend can be combined with an AI chatbot, LangGraph workflow, structured database operations, and vector-based semantic search to create an intelligent Student Database Application System.

The project covers:

* Backend development
* Database management
* AI integration
* Semantic retrieval
* API documentation
* Frontend interaction
* Testing
* Project documentation

---

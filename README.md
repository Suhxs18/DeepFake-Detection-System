# 💰 AI Finance Advisor (Django + Ollama)

An intelligent finance advisor web app built with **Django**, powered by **Ollama AI** models, and backed by **SQLite**. This application helps users get personalized investment advice based on their financial profile.

---

## 🚀 Features

- 👤 User authentication and profile management
- 📊 Financial profile creation with investment goals
- 🧠 Generates tailored investment advice using Ollama AI
- 💬 Interactive chat with AI financial advisor
- 📝 Tracks investment advice history
- 🎨 Clean, modern UI with responsive design

---

## 🛠️ Tech Stack

- **Backend:** Django (Python)
- **Database:** SQLite (Django ORM)
- **AI:** Ollama (local AI model inference)
- **Frontend:** Django Templates + Bootstrap
- **Sanitizing & Formatting:** `markdown` and `bleach`

---

## ⚙️ Installation

### 1. Clone this repository

```bash
git clone https://github.com/your-username/finance-advisor.git
cd finance-advisor
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run migrations

```bash
python manage.py migrate
```

### 4. Create a superuser (admin)

```bash
python manage.py createsuperuser
```

### 5. Start the development server

```bash
python manage.py runserver
```

### 6. Set up Ollama

Make sure you have Ollama installed and running with the financial advisor model:

```bash
ollama pull ALIENTELLIGENCE/financialadvisor
```

---

## 📝 Usage

1. Register a new account or log in
2. Create your financial profile with investment goals
3. Get personalized investment advice
4. Chat with the AI advisor for more detailed guidance

---

## 🔒 Environment Variables

Create a `.env` file in the project root with the following variables:

```
SECRET_KEY=your_django_secret_key
DEBUG=True
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=ALIENTELLIGENCE/financialadvisor
```

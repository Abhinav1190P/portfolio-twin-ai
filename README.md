# 💜 AI Digital Twin Portfolio

An AI-powered digital twin that answers questions about my professional background, projects, skills, and experience using my resume as its knowledge base.

Built with **Python**, **Gradio**, and the **Groq API**, this chatbot acts as a virtual version of me, allowing recruiters, hiring managers, and visitors to interactively explore my profile.

---

## ✨ Features

* 🤖 AI-powered digital twin
* 📄 Uses my resume as the primary knowledge source
* 💬 Natural conversational interface
* 🛠️ Function calling to collect visitor email addresses
* 📧 Email validation and duplicate prevention
* 🛡️ Prompt injection and hallucination safeguards
* ⚡ Powered by Groq's ultra-fast LLM inference
* 🎨 Modern purple-themed responsive UI
* 🚦 Built-in rate limiting and error handling

---

## 🛠️ Tech Stack

* Python
* Gradio
* Groq API
* OpenAI Python SDK
* PyPDF
* python-dotenv

---

## 📁 Project Structure

```text
.
├── app.py
├── emails.txt
├── requirements.txt
├── .env
└── twin
    ├── myresume.pdf
    └── summary.txt
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Abhinav1190P/portfolio-twin-ai.git
cd portfolio-twin-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**macOS/Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_api_key_here
```

You can obtain a free API key from Groq.

### 5. Add your personal files

Place the following files inside the `twin/` directory:

* `myresume.pdf`
* `summary.txt`

The resume provides detailed context while the summary gives the model a concise overview of your professional profile.

### 6. Run the application

```bash
python app.py
```

The application will automatically open in your browser.

---

## 💡 Example Questions

* Tell me about yourself.
* What technologies do you work with?
* Tell me about your projects.
* What experience do you have?
* How can I contact you?
* What makes you a good software engineer?

---

## 🔒 Safety Features

* Email validation
* Duplicate email detection
* API error handling
* Conversation history limiting
* Request rate limiting
* Hallucination prevention
* Prompt injection protection

---

## 📸 Preview

<img width="1703" height="850" alt="Screenshot 2026-07-10 at 6 24 57 PM" src="https://github.com/user-attachments/assets/c802c0f6-166d-4b2e-a0a3-522dce274d4d" />

---

## 🔮 Future Improvements

* RAG with vector embeddings
* Streaming responses
* Persistent database for visitor emails
* Resume download button
* Authentication
* Analytics dashboard
* Dark mode
* Multi-document knowledge base

---

## 👨‍💻 Author

**Abhinav Pandey**

Full Stack Developer passionate about AI, scalable web applications, and modern software engineering.

If you found this project interesting, feel free to connect or explore my other repositories.

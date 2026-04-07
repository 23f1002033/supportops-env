# 🚀 SupportOps-RL: Customer Support Simulation Environment

## 📌 Overview

SupportOps-RL is a real-world simulation environment for evaluating AI agents on customer support tasks.

It models:
- Customer emotions
- Trust and patience
- Churn risk
- Multi-step interactions

Built using the OpenEnv specification.

---

## 🎯 Motivation

Customer support requires:
- Handling frustrated users
- Resolving issues quickly
- Maintaining trust
- Avoiding escalation

This environment evaluates how well AI agents perform these tasks.

---

## 🧩 Tasks

| Task   | Description |
|--------|------------|
| Easy   | Order tracking |
| Medium | Refund request |
| Hard   | Angry customer |

---

## ⚙️ Action Space

Agent sends:

{
  "response": "text"
}

---

## 👀 Observation Space

Agent receives:

{
  "user_message": "text",
  "sentiment": float,
  "resolved": boolean,
  "step_count": integer
}

---

## 🧠 Hidden State

(Not visible to agent)

- trust
- patience
- churn_risk

---

## 🎯 Reward Design

Positive:
- Apology
- Taking action (refund/help)
- Efficient resolution

Negative:
- Repetition
- Asking unnecessary questions
- Long conversations

---

## 🏁 Episode Ends When

- Issue resolved
- Max steps reached
- Patience = 0

---

## 🛠️ Setup

Install dependencies:

pip install -r requirements.txt

---

## 🔑 Environment Variables

Create `.env` file:

HF_TOKEN=your_token  
API_BASE_URL=https://router.huggingface.co/v1  
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct  

---

## ▶️ Run

Test environment:

python test_env.py

Run inference:

python inference.py

---

## 📊 Output Format

[START] task=easy env=supportops model=...  
[STEP] step=1 action=... reward=0.20 done=false error=null  
[END] success=true steps=3 rewards=0.20,0.30,0.50  

---

## 🧱 Project Structure

supportops-env/
│
├── env/
│   ├── models.py
│   ├── environment.py
│   ├── grader.py
│   └── tasks/
│       ├── easy.json
│       ├── medium.json
│       └── hard.json
│
├── server.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── README.md
├── test_env.py
└── validate.py

---

## 🐳 Deployment

- Docker supported
- Hugging Face Space ready
- API endpoints:
  - /reset
  - /step

---

## 📈 Baseline Performance

Easy: 3–4 steps  
Medium: 1–2 steps  
Hard: 2–3 steps  

---

## ✨ Key Features

- Real-world simulation
- Dynamic reward system
- Hidden state modeling
- Multi-step interaction

---

## 🚀 Conclusion

SupportOps-RL provides a practical environment to evaluate AI agents in realistic customer support scenarios.
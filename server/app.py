from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from env.environment import SupportEnv
from env.models import SupportAction
from env.grader import grade
import yaml
import os

app = FastAPI(
    title="SupportOps-RL API",
    description="OpenEnv-compliant customer support simulation environment",
    version="2.0.0",
)

# ─── CORS Middleware ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SupportEnv()


class ResetRequest(BaseModel):
    task: str = "easy"


class StepRequest(BaseModel):
    response: str


# ─── Endpoints ───

@app.get("/")
def root():
    """Health check and environment info."""
    return {
        "status": "running",
        "environment": "supportops-env",
        "version": "2.0.0",
        "tasks": ["easy", "medium", "hard"],
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    """Reset the environment for a new episode."""
    task = req.task if req else "easy"
    valid_tasks = ["easy", "medium", "hard"]
    if task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Must be one of {valid_tasks}",
        )
    try:
        obs = env.reset(task=task)
        return {
            "observation": obs.model_dump(),
            "max_steps": env.max_steps,
            "task": task,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """Execute one step in the environment."""
    if not req.response.strip():
        raise HTTPException(status_code=400, detail="Response cannot be empty.")
    try:
        action = SupportAction(response=req.response)
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def get_state():
    """Inspect the current hidden state (for debugging/evaluation)."""
    try:
        state = env.state()
        return state.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/grade")
def get_grade():
    """Get the graded score for the current episode."""
    try:
        state = env.state()
        result = grade(state, max_steps=env.max_steps)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def list_tasks():
    """List all available tasks with their configurations."""
    tasks = {}
    for task_name in ["easy", "medium", "hard"]:
        try:
            task_data = env.load_task(task_name)
            tasks[task_name] = {
                "description": task_data.get("description", ""),
                "difficulty": task_data.get("difficulty", task_name),
                "max_steps": task_data.get("max_steps", 10),
                "initial_message_preview": task_data.get("initial_message", "")[:80],
            }
        except Exception:
            tasks[task_name] = {"error": "Failed to load task"}
    return {"tasks": tasks}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))

if __name__ == "__main__":
    main()
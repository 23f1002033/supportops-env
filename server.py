from fastapi import FastAPI
from pydantic import BaseModel
from env.environment import SupportEnv
from env.models import SupportAction

app = FastAPI()

env = SupportEnv()

class ResetRequest(BaseModel):
    task: str = "easy"


class StepRequest(BaseModel):
    response: str

@app.post("/reset")
def reset(req: ResetRequest):
    obs = env.reset(task=req.task)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    action = SupportAction(response=req.response)
    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/")
def root():
    return {"status": "running"}
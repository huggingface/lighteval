import os
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "helmet_data")

class HelmetTask:
    def __init__(self):
        self.prompts = {}
        self.load_prompts()

    def load_prompts(self):
        for fname in os.listdir(DATA_DIR):
            if fname.endswith(".json"):
                fpath = os.path.join(DATA_DIR, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    self.prompts[fname] = json.load(f)

    def get_prompt(self, prompt_name):
        return self.prompts.get(prompt_name)
import transformers

import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import get_models
import os

DATA_IN_TEST = 'data/in/test_data.csv'

MODEL = 'llama-70B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/predictive_without_task_and_choice/{MODEL}_unsloth/singles'


def extract_model_choice(raw_response: str) -> str:
    """
    Extracts choice from model's raw response text
    Handles common variations while maintaining strict validation
    """
    # Clean up the raw response by removing leading/trailing whitespace and quotes
    cleaned_response = raw_response.strip().strip('"')
    return cleaned_response

def format_past_trials(past_trials: list) -> str:
    """Formats past trial data for the prompt by listing choice, reward, and cumulative reward."""
    return "".join(
        f"Trial {trial['trial']}:Choice {trial['choice']}  → {trial['reward']} points ;"
        for trial in past_trials
    )

def format_past_trials_no_reward(past_trials: list) -> str:
    """Formats past trial data for the prompt by listing choice, reward, and cumulative reward."""
    return "".join(
        f"Trial {trial['trial']}:Choice {trial['choice']} ;"
        for trial in past_trials
    )

def build_slot_prompt(current_trial: int, past_trials: list, total_trials: int) -> str:
    formatted_trials = format_past_trials(past_trials)

    return f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
In this task, you have to repeatedly choose between two slot machines labeled U and P.\n
When you select one of the machines, you will win 1 or 0 points. \n
Your goal is to choose the slot machines that will give you the most points.
You will receive feedback about the outcome after making a choice.\n
The environment may change unpredictably, and past success does not guarantee future results.
You’ll need to adapt to these changes to keep finding the better machine.
You will play 1 game in total, consisting of 100 trials.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
# Task Parameters
- Game 1:
- Trial {current_trial} of {total_trials}

# History
{formatted_trials}

# Instructions
Respond with **only** one character: 'U' or 'P'. No punctuation, no quotes, no explanation.

Answer:<|start_header_id|>assistant<|end_header_id|>
"""
def build_slot_prompt_no_reward(current_trial: int, past_trials: list, total_trials: int) -> str:
    formatted_trials = format_past_trials_no_reward(past_trials)

    return f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
In this task, you have to repeatedly choose between two slot machines labeled U and P.\n
When you select one of the machines, you will win 1 or 0 points. \n
Your goal is to choose the slot machines that will give you the most points.
You will receive feedback about the outcome after making a choice.\n
The environment may change unpredictably, and past success does not guarantee future results.
You’ll need to adapt to these changes to keep finding the better machine.
You will play 1 game in total, consisting of 100 trials.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
# Task Parameters
- Game 1:
- Trial {current_trial} of {total_trials}

# History
{formatted_trials}

# Instructions
Respond with **only** one character: 'U' or 'P'. No punctuation, no quotes, no explanation.

Answer:<|start_header_id|>assistant<|end_header_id|>
"""

def simulate_participant(df_participant: pd.DataFrame, model, tokenizer, letter_token_ids, build_slot_prompt):
    """Simulates a participant with log-likelihood tracking"""
    history = []
    cumulative_reward = 0
    total_trials = len(df_participant)


    for trial in range(total_trials):
        row = df_participant.iloc[trial]
        trial_num = row['trial']
        human_choice = row['choice']
        reward = row['reward']
        cumulative_reward += reward

        # Build prompt using actual human history
        past_trials = []
        for past_idx in range(trial):
            past_row = df_participant.iloc[past_idx]
            past_trials.append({
                "trial": past_row['trial'],
                "choice": past_row['choice'],
                "reward": past_row['reward'],
                "cumulative_reward": df_participant.iloc[:past_idx+1]['reward'].sum()
            })

        prompt = build_slot_prompt(trial_num, past_trials, total_trials)

         # --- Run model on prompt ---
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0, -1]
        probs = F.softmax(logits, dim=-1)
        pred_token_id = torch.argmax(probs).item()
        # Top-k tokens
        # Top-2 tokens and their probabilities
        topk = torch.topk(probs, k=2)
        top2_tokens = []
        for idx, prob in zip(topk.indices, topk.values):
            decoded_token = tokenizer.decode(idx.item()).strip()
            top2_tokens.append({
                "token": decoded_token,
                "prob": prob.item()
            })


        # --- Map to model choice (U or P) ---
        token_id_to_letter = {v: k for k, v in letter_token_ids.items()}
        model_choice = token_id_to_letter.get(pred_token_id, "INVALID")
        # --- Log-likelihood of human’s actual choice ---
        log_likelihood = None
        if human_choice in letter_token_ids:
            user_token_id = letter_token_ids[human_choice]
            log_likelihood = torch.log(probs[user_token_id] + 1e-8).item() # Fixed: Use token ID for indexing

        # --- Get the actual reward (from human data) ---
        # The reward is already taken from the dataframe at the beginning of the loop
        # reward = df.loc[df["trial"] == trial, "reward"].values[0]
        # cumulative_reward += reward # Cumulative reward is updated earlier

        history.append({
            "trial": trial,
            "prompt": prompt,
            "model_choice": model_choice,
            "human_choice": human_choice,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "log_likelihood": log_likelihood,
            "top2_tokens": top2_tokens
        })
        print(f"Trial {trial}: Human {human_choice}, Model {model_choice}, LL: {log_likelihood}")

    return pd.DataFrame(history)



def main():

    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)

    model, tokenizer = get_models.get_model_no_pipe_unsloth(MODEL)
    timeline = pd.read_csv(DATA_IN_TEST)
    timeline['choice'] = timeline['choice'].map({0: 'U', 1: 'P'})
    model_ids = timeline['model_id'].unique()


    letter_token_ids = {
    "U": tokenizer("U", add_special_tokens=False)['input_ids'][0],
    "P": tokenizer("P", add_special_tokens=False)['input_ids'][0],
}


    for model_id in model_ids:
        print(f"\n🧠 Simulating model {model_id}")
        out_path = f'{DATA_FOLDER_OUT}/model_' + str(model_id) + '.csv'

        if os.path.exists(out_path):
            print(f"Model {model_id} already simulated. Skipping...")
            continue

        # Run simulation with model and tokenizer passed
        model_data = timeline[timeline['model_id'] == model_id]
        result = simulate_participant(model_data, model, tokenizer, letter_token_ids, build_slot_prompt_no_reward)
        result.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()

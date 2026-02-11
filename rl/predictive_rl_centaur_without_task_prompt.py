import transformers

import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import get_models
import os
import gc

DATA_IN_TEST = 'data/in/test_data.csv'

MODEL = 'centaur-70B-adapter'
DATA_FOLDER_OUT = f'data/out/predictive_without_task_and_choice/{MODEL}/singles'

def generate_seeds(num_seeds=20, seed=42):
    """Generates a list of random seeds.

    Args:
        num_seeds: The number of seeds to generate.
        seed: The initial seed for the random number generator (for reproducibility).

    Returns:
        A list of random integer seeds.
    """
    random.seed(seed)  # Set initial seed for reproducibility
    seeds = [random.randint(1, 100000) for _ in range(num_seeds)]
    return seeds




def extract_model_choice(raw_response: str) -> str:
    """
    Extracts choice from model's raw response text
    Handles common variations while maintaining strict validation
    """
    # Clean up the raw response by removing leading/trailing whitespace and quotes
    cleaned_response = raw_response.strip().strip('"')
    return cleaned_response

#add prompt without task instruction
def build_slot_prompt_without_instruction(past_trials: list, total_trials: int) -> str:
    """Builds the prompt for the current trial with past trial data."""
    recent_trials = past_trials
    prompt = (
              "In this task, you have to repeatedly choose between two slot machines labeled U and P.\n"
              "You can choose a slot machine by pressing its corresponding key."
    )

    # Add history of past trials to the prompt
    for past_trial in recent_trials:
        prompt += f"You press <<{past_trial['choice']}>>.\n"

    # Add the current choice prompt
    prompt += f"You press <<"
    return prompt

#add prompt with last trial
def build_slot_prompt_last_trial(past_trials: list, total_trials: int) -> str:
    """Builds the prompt for the current trial with past trial data."""
    recent_trials = past_trials[-1:]
    prompt = (
              "In this task, you have to repeatedly choose between two slot machines labeled U and P.\n"
              "You can choose a slot machine by pressing its corresponding key."
              "When you select one of the machines, you will win 1 or 0 points."
              "Your goal is to choose the slot machines that will give you the most points."
              "You will receive feedback about the outcome after making a choice.\n"
              "The environment may change unpredictably, and past success does not guarantee future results. You’ll need to adapt to these changes to keep finding the better machine."
              f"You will play 1 game in total, consisting of {total_trials} trials."
            f" Game 1:"
    )
    # Add history of past trials to the prompt
    for past_trial in recent_trials:
        prompt += f"You press <<{past_trial['choice']}> and get {past_trial['reward']} points.\n"
    # Add the current choice prompt
    prompt += f"You press <<"
    return prompt

#add zeroshot prompt
def build_slot_prompt_zeroshot(past_trials: list, total_trials: int) -> str:
    """Builds the prompt for the current trial with past trial data."""
    recent_trials = past_trials
    prompt = (
              "In this task, you have to repeatedly choose between two slot machines labeled U and P.\n"
              "You can choose a slot machine by pressing its corresponding key."
              "When you select one of the machines, you will win 1 or 0 points."
              "Your goal is to choose the slot machines that will give you the most points."
              "You will receive feedback about the outcome after making a choice.\n"
              "The environment may change unpredictably, and past success does not guarantee future results. You’ll need to adapt to these changes to keep finding the better machine."
              f"You will play 1 game in total, consisting of {total_trials} trials."
            f" Game 1:"
    )
    # Add the current choice prompt
    prompt += f"You press <<"
    return prompt

def build_slot_prompt_no_reward(past_trials: list, total_trials: int) -> str:
    """Builds the prompt for the current trial with past trial data."""
    recent_trials = past_trials
    prompt = (
              "In this task, you have to repeatedly choose between two slot machines labeled U and P.\n"
              "You can choose a slot machine by pressing its corresponding key."
              "When you select one of the machines, you will win 1 or 0 points."
              "Your goal is to choose the slot machines that will give you the most points."
              "You will receive feedback about the outcome after making a choice.\n"
              "The environment may change unpredictably, and past success does not guarantee future results. You’ll need to adapt to these changes to keep finding the better machine."
              f"You will play 1 game in total, consisting of {total_trials} trials."
            f" Game 1:"
    )
    # Add history of past trials to the prompt
    for past_trial in recent_trials:
        prompt += f"You press <<{past_trial['choice']}>>.\n"
    # Add the current choice prompt
    prompt += f"You press <<"
    return prompt

def fix_seed(seed: int):
    """Fixes the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    transformers.set_seed(seed)  # For Hugging Face models


def simulate_participant(df_participant: pd.DataFrame, model, tokenizer, letter_token_ids):
    """Simulates a participant with log-likelihood tracking, NLL calculations, and top-2 token probabilities"""
    history = []
    cumulative_reward = 0
    total_trials = len(df_participant)
    print(f"Total trials: {total_trials}")

    # Build the prompt once using all trials
    past_trials = []
    for trial in range(total_trials):
        row = df_participant.iloc[trial]
        past_trials.append({
            "trial": row['trial'],
            "choice": row['choice'],
            "reward": row['reward'],
            "cumulative_reward": df_participant.iloc[:trial+1]['reward'].sum()
        })

    prompt = build_slot_prompt(past_trials, total_trials)

    # Tokenize the full participant prompt once
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32768)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    per_trial_results = []
    all_individual_token_nlls = []

    with torch.no_grad():
        # Single forward pass for the entire participant prompt
        outputs = model(**inputs)
        logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

        # Extract NLL for each choice token progressively
        choice_patterns = [
            r'You press <<([^>]+)>>',
            r'<<([^>]+)>>'  # General fallback pattern
        ]

        choice_matches = []
        for pattern in choice_patterns:
            matches = list(re.finditer(pattern, prompt))
            if matches:
                choice_matches = matches
                break

        if not choice_matches:
            print("Warning: No choices found in the prompt, skipping...")
            return per_trial_results, float('inf')

        for choice_idx, match in enumerate(choice_matches):
            choice_value = match.group(1)  # The content inside <<>>
            choice_start_pos = match.start(1)

            # Tokenize text up to the choice to find token position
            text_before_choice = prompt[:choice_start_pos]
            tokens_before = tokenizer(text_before_choice, return_tensors="pt", truncation=False)
            choice_start_token_pos = len(tokens_before['input_ids'][0]) - 1

            # Tokenize just the choice content
            choice_tokens = tokenizer(choice_value, return_tensors="pt", add_special_tokens=False)
            choice_token_ids = choice_tokens['input_ids'][0]

            # Extract logits for predicting the choice tokens
            if choice_start_token_pos + len(choice_token_ids) <= len(logits):
                choice_logits = logits[choice_start_token_pos:choice_start_token_pos + len(choice_token_ids)]

                # Compute NLL for each choice token and extract top-2 tokens with probabilities
                individual_token_nlls = []
                top2_tokens_probs = []
                for i, token_id in enumerate(choice_token_ids):
                    if i < len(choice_logits):
                        token_logits = choice_logits[i]
                        log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
                        token_nll = -log_probs[token_id].item()
                        individual_token_nlls.append(token_nll)

                        # Get top-2 tokens and their probabilities
                        top2_probs, top2_indices = torch.topk(log_probs, 2)
                        top2_tokens = tokenizer.convert_ids_to_tokens(top2_indices.tolist())
                        top2_probs = top2_probs.exp().tolist()  # Convert log probs to probabilities
                        top2_tokens_probs.append({
                            "top_tokens": top2_tokens,
                            "top_probs": top2_probs
                        })

                # Aggregate NLL
                if individual_token_nlls:
                    choice_nll = sum(individual_token_nlls) / len(individual_token_nlls)
                    all_individual_token_nlls.extend(individual_token_nlls)
                else:
                    choice_nll = float('inf')
            else:
                choice_nll = float('inf')
                top2_tokens_probs = []

            # Store trial result
            trial_result = {
                'trial_index': choice_idx,
                'ground_truth_choice': choice_value,
                'trial_nll': choice_nll,
                'has_history': choice_idx > 0,
                'num_history_choices': choice_idx,
                'top2_tokens_probs': top2_tokens_probs
            }
            per_trial_results.append(trial_result)

    # Compute summary statistics
    valid_trial_nlls = [r['trial_nll'] for r in per_trial_results if r['trial_nll'] != float('inf')]
    overall_nll = sum(valid_trial_nlls) / len(valid_trial_nlls) if valid_trial_nlls else float('inf')

    print(f"✅ Simulation complete")
    print(f"🎯 Overall NLL: {overall_nll:.4f}")
    print(f"📊 Total tokens evaluated: {len(all_individual_token_nlls)}")

    return per_trial_results, overall_nll

def simulate_participant_trial_wise(df_participant: pd.DataFrame, model, tokenizer,letter_token_ids,build_slot_prompt):
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

        prompt = build_slot_prompt(past_trials, total_trials)

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
    valid_lls = [item['log_likelihood'] for item in history if item['log_likelihood'] is not None]
    overall_nll = -sum(valid_lls) / len(valid_lls) if valid_lls else float('inf')
    return pd.DataFrame(history),overall_nll



def main():

    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)

    model, tokenizer = get_models.get_model_no_pipe_unsloth(MODEL)
    timeline = pd.read_csv(DATA_IN_TEST)
    timeline['choice'] = timeline['choice'].map({0: 'U', 1: 'P'})
    overall_nlls = []
    model_ids = timeline['model_id'].unique()
    letter_token_ids = {
    "U": tokenizer("U", add_special_tokens=False)['input_ids'][0],
    "P": tokenizer("P", add_special_tokens=False)['input_ids'][0],
}
    test_cases_no_reward=['no_reward']
    for test in test_cases_no_reward:
        if test == 'zero-shot':
            build_slot_prompt = build_slot_prompt_zeroshot
        elif test == 'last-trial':
            build_slot_prompt = build_slot_prompt_last_trial
        elif test == 'without_task_prompt':
            build_slot_prompt = build_slot_prompt_without_instruction
        elif test == 'no_reward':
            build_slot_prompt = build_slot_prompt_no_reward
    for model_id in model_ids:
        print(f"\n🧠 Simulating model {model_id}")
        out_path = f'{DATA_FOLDER_OUT}/model_' + str(model_id) + '.csv'

        if os.path.exists(out_path):
            print(f"Model {model_id} already simulated. Skipping...")
            continue

        # Run simulation with model and tokenizer passed
        model_data = timeline[timeline['model_id'] == model_id]
        trial_results,overall_nll=simulate_participant_trial_wise(model_data, model , tokenizer,letter_token_ids,build_slot_prompt)
        overall_nlls.append(overall_nll)
        # Save results
        df_results = pd.DataFrame(trial_results)
        df_results.to_csv(out_path, index=False)

        print(f"Participant {model_id} - Overall NLL: {overall_nll:.4f}")
    # Summary of overall NLLs
    if overall_nlls:
        avg_nll = np.mean(overall_nlls)
        std_nll = np.std(overall_nlls)
        print(f"\n📊 Summary of Overall NLLs across all models:")
        print(f"Average NLL: {avg_nll:.4f}, Std Dev: {std_nll:.4f}")
    
if __name__ == "__main__":
    main()

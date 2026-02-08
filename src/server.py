import torch

def aggregate(global_model, updates):
    new_state = global_model.state_dict()

    for key in new_state:
        avg = sum(update[key] for update in updates) / len(updates)
        new_state[key] += avg

    global_model.load_state_dict(new_state)
    return global_model

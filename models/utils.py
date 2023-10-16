import copy
def load_sd_for_matched_keys(model, sd, ignore_keys=[]):
    model_sd = copy.deepcopy(model.state_dict())
    for param in model_sd.keys():
        if param in sd and param not in ignore_keys:
            try:
                model_sd[param] = sd[param]
            except Exception as e:
                print(e)
                print(f'failed to load {param}')
                
    model.load_state_dict(model_sd)
    return model
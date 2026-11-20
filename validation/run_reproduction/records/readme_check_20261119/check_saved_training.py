import json
import pickle
import types
from pathlib import Path
import torch
from safetensors import safe_open

class SavedConfig:
    """Read the serialized configuration fields without importing generated code."""
    pass

class ConfigUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'UnslothSFTTrainer' and name == 'UnslothSFTConfig':
            return SavedConfig
        return super().find_class(module, name)

reader = types.ModuleType('config_pickle')
reader.__dict__.update(pickle.__dict__)
reader.Unpickler = ConfigUnpickler
root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
for task,folder,prefix in [('se','Structure elucidation','SE'),('ncsp','NMR chemical shift prediction','NCSP')]:
    trained = out/task/'training_fixed/final_model'
    if not (trained/'training_args.bin').exists():
        continue
    args = torch.load(trained/'training_args.bin', weights_only=False, map_location='cpu', pickle_module=reader)
    fields = ['max_length','dataset_num_proc','dataset_text_field','learning_rate','per_device_train_batch_size','gradient_accumulation_steps','num_train_epochs','warmup_steps']
    result = {key:getattr(args,key,None) for key in fields}
    assert result['max_length'] == 5000 and result['dataset_num_proc'] == 1, result
    original = root/'weight'/folder/f'{prefix}-CoT-1_MultiNMR_real200k_weight/model.safetensors'
    key = 'model.layers.0.self_attn.q_proj.weight'
    with safe_open(str(original),framework='pt',device='cpu') as a, safe_open(str(trained/'model.safetensors'),framework='pt',device='cpu') as b:
        delta = (a.get_tensor(key).float()-b.get_tensor(key).float()).abs()
    result.update(tensor=key,changed_elements=int((delta>0).sum()),max_weight_change=float(delta.max()))
    assert result['changed_elements'] > 0
    state = json.loads((out/task/'training_fixed/outputs/checkpoint-2/trainer_state.json').read_text())
    result['optimizer_steps'] = state['global_step']
    result['log_history'] = state['log_history']
    assert result['optimizer_steps'] == 2
    (out/task/'fixed_training_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(task, result)

import torch
import torch.nn.functional as F
from torch.linalg import svd
from sklearn.cluster import KMeans
import random
from tqdm import tqdm
from transformers import Trainer



def init_svd(proj_weight, compressed_dim_per_head, num_heads):
    """
    proj: k_proj or v_proj, weight.shape = [hidden_size, self.num_heads * head_dim].T
    compressed_dim_per_head: eg. 64
    num_heads: eg. 32
    return: proj_a_weight, shape = [hidden_size, self.num_heads * compressed_dim].T; 
            proj_b_weight, shape = [self.num_heads * compressed_dim, hidden_size].T
    """
    orig_dtype = proj_weight.dtype
    compressed_dim = compressed_dim_per_head * num_heads
    u, s, vt = svd(proj_weight.T.to(torch.float32), full_matrices=False)   # u.shape = [m, n], s.shape = [n], vt.shape = [n, n]
    u = u[:, :compressed_dim]
    s = s[ :compressed_dim]
    vt = vt[ :compressed_dim, :]
    
    proj_a_weight = (u @ torch.sqrt(torch.diag(s))).T.to(orig_dtype)
    proj_b_weight = (torch.sqrt(torch.diag(s)) @ vt).T.to(orig_dtype)

    return proj_a_weight, proj_b_weight

def init_asvd(proj_weight, calib_data, compressed_dim_per_head, num_heads, alpha=0.5):
    orig_dtype = proj_weight.dtype
    compressed_dim = compressed_dim_per_head * num_heads
    calib_data += 1e-6
    calib_data = (calib_data ** alpha).to(orig_dtype)
    scaled_proj_weight = torch.diag(calib_data) @ proj_weight.T
    u, s, vt = svd(scaled_proj_weight.to(torch.float32), full_matrices=False)
    u = u[:, :compressed_dim]
    s = s[ :compressed_dim]
    vt = vt[ :compressed_dim, :]
    
    u = torch.diag(calib_data**(-1)).to(torch.float32) @ u

    proj_a_weight = (u @ torch.sqrt(torch.diag(s))).T.to(orig_dtype)
    proj_b_weight = (torch.sqrt(torch.diag(s)) @ vt).T.to(orig_dtype)
    
    return proj_a_weight, proj_b_weight

def init_svd_parallel_inference(proj_weight, compressed_dim_per_head, num_heads):
    """
    proj: k_proj or v_proj, weight.shape = [hidden_size, self.num_heads * head_dim].T
    compressed_dim_per_head: eg. 64
    num_heads: eg. 32
    return: proj_a_weight, shape = [hidden_size, head_dim + self.num_heads * compressed_dim].T; 
            proj_b_weight, shape = [self.num_heads * compressed_dim, hidden_size].T
    """
    orig_dtype = proj_weight.dtype
    compressed_dim = compressed_dim_per_head * num_heads
    u, s, vt = svd(proj_weight.T.to(torch.float32), full_matrices=False)   # u.shape = [m, n], s.shape = [n], vt.shape = [n, n]
    u = u[:, :compressed_dim]
    s = s[ :compressed_dim]
    vt = vt[ :compressed_dim, :]
    
    proj_a_weight = torch.cat([proj_weight.T, (u @ torch.sqrt(torch.diag(s))).to(orig_dtype)], dim=1).T
    proj_b_weight = (torch.sqrt(torch.diag(s)) @ vt).T.to(orig_dtype)

    return proj_a_weight, proj_b_weight

def init_asvd_parallel_inference(proj_weight, calib_data, compressed_dim_per_head, num_heads, alpha=0.5):
    """
    calib_data.shape = [4096]
    """
    orig_dtype = proj_weight.dtype
    compressed_dim = compressed_dim_per_head * num_heads
    calib_data += 1e-6
    calib_data = calib_data ** alpha
    scaled_proj_weight = torch.diag(calib_data) @ proj_weight.T
    u, s, vt = svd(scaled_proj_weight.to(torch.float32), full_matrices=False)
    u = u[:, :compressed_dim]
    s = s[ :compressed_dim]
    vt = vt[ :compressed_dim, :]

    u = torch.diag(calib_data**(-1)).to(torch.float32) @ u

    proj_a_weight = torch.cat([proj_weight.T, (u @ torch.sqrt(torch.diag(s))).to(orig_dtype)], dim=1).T
    proj_b_weight = (torch.sqrt(torch.diag(s)) @ vt).T.to(orig_dtype)

    return proj_a_weight, proj_b_weight

def init_svd_parallel_inference_load_ckpt(proj_weight, compressor_weight_a, compressor_weight_b):
    """
    proj: k_proj or v_proj, weight.shape = [hidden_size, self.num_heads * head_dim].T
    compressed_weight_a.shape = [hidden_size, compressed_dim].T
    compressed_weight_b.shape = [compressed_dim, hidden_size].T

    return: proj_a_weight, shape = [hidden_size, head_dim + self.num_heads * compressed_dim].T; 
            proj_b_weight, shape = [self.num_heads * compressed_dim, hidden_size].T
    """

    proj_a_weight = torch.cat([proj_weight.T, compressor_weight_a.T], dim=1).T
    proj_b_weight = compressor_weight_b

    return proj_a_weight, proj_b_weight


def save_kvcompressor_ckpt(model, ckpt_save_path, kv, args=None):
    """
    saved format: [k_model_layer.state_dict() for k_model_layer in k_models]
        if svd_init: [(k_proj_a.state_dict(), k_proj_b.state_dict()) for k_model_layer in k_models]
    kv: 'k' or 'v'
    """
    state_dicts = []

    # replace the original attention modules
    if 'llama' in model.config.architectures[0].lower() or 'vicuna' in model.config.architectures[0].lower() or 'longchat' in model.config.architectures[0].lower() or 'baichuan' in model.config.architectures[0].lower():
        for i in tqdm(range(len(model.model.layers))):
            block = model.model.layers[i]
            state_dict = (block.self_attn.k_proj_a.state_dict(), block.self_attn.k_proj_b.state_dict()) if kv == 'k' else (block.self_attn.v_proj_a.state_dict(), block.self_attn.v_proj_b.state_dict())
            state_dicts.append(state_dict)
    else:
        raise NotImplementedError(f"Not support model architecture: {model.config.architectures[0]}")

    with open(ckpt_save_path, 'wb') as f:
        torch.save(state_dicts, f)
    
### Trainers
class TrainerForSVDKV(Trainer):
    def __init__(self, model, train_dataset, args, data_collator):
        super().__init__(model=model, train_dataset=train_dataset, args=args, data_collator=data_collator)
        self.kvmse_lambda = 1.0 # do not use language modeling loss
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # collect local kvmse loss
        attention_ops = []
        for name, module in model.named_modules():
            if hasattr(module, "k_mse_loss"):
                attention_ops.append(module)
        assert len(attention_ops) == model.config.num_hidden_layers
        loss_mse = 0.0
        for module in attention_ops:
            loss_mse += (module.k_mse_loss + module.v_mse_loss)
        
        loss = self.kvmse_lambda * loss_mse + (1.0 - self.kvmse_lambda) * loss

        return (loss, outputs) if return_outputs else loss

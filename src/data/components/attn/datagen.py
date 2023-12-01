import os
import numpy as np

import torch


def Attn_generate(
    data_dir,
    size,
    seq_length,
    input_dim,
    Gaussian_input=False,
):
    r""" TODO
    """

    if data_dir is not None:
        input_file_path = data_dir + f"attn_T{seq_length}_N{size}_D{input_dim}_inputs.npy"
        output_file_path = data_dir + f"attn_T{seq_length}_N{size}_D{input_dim}_outputs.npy"

        # Skip generation if files exist
        if os.path.isfile(output_file_path) and os.path.isfile(input_file_path):
            print("Files for attn already exist, skipping generation.")
            inputs = np.load(input_file_path)
            outputs = np.load(output_file_path)
        else:
            inputs = np.random.normal(size=(size, seq_length, input_dim))

            # Make input (continuous) Gaussian process
            if Gaussian_input:
                inputs = np.cumsum(inputs, axis=1)

            # Make into tensor
            inputs = torch.from_numpy(inputs).float()

            outputs = torch.nn.functional.scaled_dot_product_attention(
                inputs, inputs, inputs, 
                attn_mask=None, 
                dropout_p=0.0, 
                is_causal=True, 
                # scale=None, # Supported from 2.1
                )
            
            inputs = inputs.numpy()
            outputs = outputs.numpy()

            np.save(input_file_path, inputs)
            np.save(output_file_path, outputs)

        print("Attn_generate done")

# Now we apply a square linear transform to make the attention harder. 
def QKVAttn_generate(
    data_dir,
    size,
    seq_length,
    input_dim,
    Gaussian_input=False,
):
    r""" TODO
    """

    if data_dir is not None:
        input_file_path = data_dir + f"qkvattn_T{seq_length}_N{size}_D{input_dim}_inputs.npy"
        output_file_path = data_dir + f"qkvattn_T{seq_length}_N{size}_D{input_dim}_outputs.npy"

        # Skip generation if files exist
        if os.path.isfile(output_file_path) and os.path.isfile(input_file_path):
            print("Files for attn already exist, skipping generation.")
            inputs = np.load(input_file_path)
            outputs = np.load(output_file_path)
        else:
            inputs = np.random.normal(size=(size, seq_length, input_dim))

            # Make input (continuous) Gaussian process
            if Gaussian_input:
                inputs = np.cumsum(inputs, axis=1)

            # Make into tensor
            inputs = torch.from_numpy(inputs).float()
            q = torch.nn.Linear(input_dim, input_dim)
            k = torch.nn.Linear(input_dim, input_dim)
            v = torch.nn.Linear(input_dim, input_dim)

            outputs = torch.nn.functional.scaled_dot_product_attention(
                q(inputs), k(inputs), v(inputs), 
                attn_mask=None, 
                dropout_p=0.0, 
                is_causal=True, 
                # scale=None, # Supported from 2.1
                )
            
            inputs = inputs.numpy()
            outputs = outputs.detach().numpy()

            np.save(input_file_path, inputs)
            np.save(output_file_path, outputs)

        print("QKVAttn_generate done")

if __name__ == "__main__":
    pass

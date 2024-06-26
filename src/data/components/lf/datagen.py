import os

import numpy as np
import scipy


def manual_convolve(inputs, rho, dt):
    """_summary_

    Args:
        inputs (_type_): _description_
        rho (_type_): _description_
        dt (_type_): _description_

    Returns:
        _type_: _description_
    """
    seq_length, _ = inputs.shape[1], inputs.shape[2]

    outputs = []
    for t in range(seq_length):
        output = 0
        for s in range(t + 1):
            output += inputs[:, t - s, :] * rho(s * dt)
        outputs.append(output)
    direct_outputs = np.array(outputs).transpose(1, 0, 2)
    return direct_outputs


def fft_convolve(inputs, rho, dt):
    """_summary_

    Args:
        inputs (_type_): _description_
        rho (_type_): _description_
        dt (_type_): _description_

    Returns:
        _type_: _description_
    """
    # inputs: B * T * D
    seq_length, input_dim = inputs.shape[1], inputs.shape[2]

    # Create the rho values for the entire sequence length
    rho_vals = np.array([rho(t * dt) for t in range(seq_length)])

    # Pad the sequences to avoid circular convolution
    padded_rho = np.concatenate((rho_vals, np.zeros_like(rho_vals)))
    padded_inputs = np.concatenate(
        (inputs, np.zeros((inputs.shape[0], seq_length, input_dim))), axis=1
    )

    # FFT
    rho_fft = np.fft.fft(padded_rho)
    inputs_fft = np.fft.fft(padded_inputs, axis=1)

    # Element-wise multiplication in frequency domain
    result_fft = inputs_fft * rho_fft[np.newaxis, :, np.newaxis]

    # Inverse FFT
    conv_result = np.fft.ifft(result_fft, axis=1)

    # Return the result up to seq_length
    return np.real(conv_result[:, :seq_length, :])


def LF_generate(
    data_dir,
    size,
    seq_length,
    input_dim,
    dt,
    rho,
    rho_name,
    Gaussian_input=False,
):
    r"""Dataset generation for linear functional.

    H_t(x) = \int_{0}^t rho(t-s) x(s) ds
    """

    if data_dir is not None:
        input_file_path = data_dir + f"lf_{rho_name}_T{seq_length}_N{size}_dt{dt}_inputs.npy"
        output_file_path = data_dir + f"lf_{rho_name}_T{seq_length}_N{size}_dt{dt}_outputs.npy"

        # Skip generation if files exist
        if os.path.isfile(output_file_path) and os.path.isfile(input_file_path):
            print("Files for lf already exist, skipping generation.")
            inputs = np.load(input_file_path)
            outputs = np.load(output_file_path)
        else:
            inputs = dt * np.random.normal(size=(size, seq_length, input_dim))

            # Make input Gaussian process
            if Gaussian_input:
                inputs = np.cumsum(inputs, axis=1)

            output_reshaped = fft_convolve(inputs, rho, dt)  # From O(T^2) to O(T logT)

            np.save(input_file_path, inputs)
            np.save(output_file_path, output_reshaped)

        # Normalize
        # inputs /= np.max(np.abs(inputs))
        # outputs /= np.max(np.abs(outputs))

        print("LF_generate done")
        # print("In lf datagen, input shape", inputs.shape)
        # print("In lf datagen, output shape", output_reshaped.shape)


# TODO
# Code refactor with lf_datamodule into some new memory register
rhos = {
    "exp": lambda t: np.exp(-t),
    "pol": lambda t: 1 / (1 + 0.1 * t) ** 1.1,
    "sin": lambda t: np.sin(t),
    "shift": lambda t: 1 if (t > 9) and (t < 11) else 0,
    "twoparts": lambda t: np.exp(-t) + np.exp(-((t - 9.0) ** 2)),
    "airy": lambda t: scipy.special.airy(t - 6.0)[0],
}

if __name__ == "__main__":
    pass

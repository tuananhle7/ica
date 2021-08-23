import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import util
import numpy as np
import ica
import numpyro
import math


def generate_signal(key, num_samples):
    """Generate the toy signal from Aapo's tutorial
    Args
        key
        num_samples
    
    Returns
        true_source [num_samples, 2]
        mixing_matrix [2, 2]
        signal [num_samples, 2]
    """
    dim = 2

    # Source
    low, high = jnp.full((dim,), -math.sqrt(3)), jnp.full((dim,), math.sqrt(3))
    true_source_dist = numpyro.distributions.Independent(
        numpyro.distributions.Uniform(low, high), reinterpreted_batch_ndims=1
    )
    true_source = true_source_dist.sample(key, (num_samples,))

    # Mixing matrix
    mixing_matrix = jnp.array([[2, 3], [2, 1]])

    # Observed signal
    signal = jax.vmap(ica.get_signal, (None, 0), 0)(mixing_matrix, true_source)

    return true_source, mixing_matrix, signal


def main():
    num_iterations = 500
    num_samples = 1000
    lr = 1e-3

    # Generate signal
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    true_source, mixing_matrix, signal = generate_signal(subkey, num_samples)

    # Run ICA
    key, subkey = jax.random.split(key)
    total_log_likelihoods, raw_mixing_matrices, preprocessing_params = ica.ica(
        key, signal, ica.get_subgaussian_log_prob, num_iterations=num_iterations, lr=lr
    )

    # Plot likelihoods
    fig, ax = plt.subplots(1, 1)
    ax.plot(total_log_likelihoods)
    util.save_fig(fig, "save/toy_data/total_log_likelihoods.png")

    # Print mixing matrix
    A, mean = preprocessing_params
    signal_preprocessed, _ = ica.preprocess_signal(signal)
    print(f"Mixing matrix: {jnp.linalg.inv(A) @ ica.get_mixing_matrix(raw_mixing_matrices[-1])}")
    print(f"True mixing matrix: {mixing_matrix}")

    # Plot training progress
    # --Plot pngs
    scatter_kwargs = {"s": 0.5, "c": "black", "marker": "o"}
    img_paths = []
    for iteration in np.linspace(0, num_iterations, 11):
        source = jax.vmap(ica.get_source, (0, None), 0)(
            signal_preprocessed, raw_mixing_matrices[int(iteration)]
        )

        fig, axss = plt.subplots(2, 2, sharex=True, sharey=True)
        axss[0, 0].scatter(true_source[:, 0], true_source[:, 1], **scatter_kwargs)
        axss[0, 0].set_title("True source")
        axss[0, 1].scatter(signal[:, 0], signal[:, 1], **scatter_kwargs)
        axss[0, 1].set_title("Observed signal")
        axss[1, 0].scatter(signal_preprocessed[:, 0], signal_preprocessed[:, 1], **scatter_kwargs)
        axss[1, 0].set_title("PCA of observed signal")
        axss[1, 1].scatter(source[:, 0], source[:, 1], **scatter_kwargs)
        axss[1, 1].set_title("Recovered source")

        # Formatting
        for ax in axss.flat:
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.tick_params(direction="in")

        # Title
        fig.suptitle(f"Iteration {int(iteration)}")

        # Save
        img_path = f"save/toy_data/optinization/{int(iteration)}.png"
        util.save_fig(
            fig, img_path, dpi=400, tight_layout_kwargs={"rect": [0, 0, 1, 1.02]},
        )
        img_paths.append(img_path)

    # --Make gif
    util.make_gif(img_paths, "save/toy_data/optimization.gif", 10)


if __name__ == "__main__":
    main()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import util
import numpy as np
import ica


def generate_signal(key, num_timesteps=500):
    """Create a timeseries signal that's used in Aapo's tutorial.

    Based on
    https://github.com/probml/pyprobml/blob/master/scripts/ica_demo.py#L23

    Args
        key
        num_timesteps
    
    Returns
        true_source [num_timesteps, 4]
        mixing_matrix [4, 4]
        signal [num_timesteps, 4]
    """
    timesteps = jnp.arange(0, num_timesteps)
    dim = 4
    true_source = jnp.zeros((num_timesteps, dim))

    true_source = true_source.at[:, 0].set(jnp.sin(timesteps / 2))  # sinusoid
    true_source = true_source.at[:, 1].set(((timesteps % 23 - 11) / 9) ** 5)
    true_source = true_source.at[:, 2].set((timesteps % 27 - 13) / 9)  # sawtooth

    key, subkey = jax.random.split(key)
    rand = jax.random.uniform(subkey, (num_timesteps,))
    key, subkey = jax.random.split(key)
    true_source = true_source.at[:, 3].set(
        jnp.where(rand < 0.5, rand * 2 - 1, -1)
        * jnp.log(jax.random.uniform(subkey, (num_timesteps,)))
    )  # impulsive noise

    true_source /= true_source.std(axis=0)
    true_source -= true_source.mean(axis=0)
    key, subkey = jax.random.split(key)
    mixing_matrix = jax.random.uniform(subkey, (dim, dim))  # mixing matrix
    return (
        true_source,
        mixing_matrix,
        jax.vmap(ica.get_signal, (None, 0), 0)(mixing_matrix, true_source),
    )


def plot_timeseries_signal(axs, signal):
    """
    Args
        axs (list of length 4)
        signal [num_timesteps, 4]
    """
    num_timesteps, dim = signal.shape
    for i in range(dim):
        axs[i].plot(signal[:, i], color="black", linewidth=0.7)


def main():
    num_iterations = 500
    lr = 1e-3

    # Generate signal
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    true_source, mixing_matrix, signal = generate_signal(subkey)

    # Run ICA
    key, subkey = jax.random.split(key)
    total_log_likelihoods, raw_mixing_matrices, preprocessing_params = ica.ica(
        key, signal, ica.get_supergaussian_log_prob, num_iterations=num_iterations, lr=lr
    )

    # Plot likelihoods
    fig, ax = plt.subplots(1, 1)
    ax.plot(total_log_likelihoods)
    util.save_fig(fig, "save/timeseries/total_log_likelihoods.png")

    # Print mixing matrix
    A, mean = preprocessing_params
    signal_preprocessed, _ = ica.preprocess_signal(signal)
    print(f"Mixing matrix: {jnp.linalg.inv(A) @ ica.get_mixing_matrix(raw_mixing_matrices[-1])}")
    print(f"True mixing matrix: {mixing_matrix}")

    # Plot training progress
    # --Plot pngs
    img_paths = []
    for iteration in np.linspace(0, num_iterations, 11):
        source = jax.vmap(ica.get_source, (0, None), 0)(
            signal_preprocessed, raw_mixing_matrices[int(iteration)]
        )

        fig, axss = plt.subplots(8, 2, sharex=True, sharey=True, figsize=(12, 12))
        plot_timeseries_signal(axss[:4, 0], true_source)
        axss[0, 0].set_title("True source")
        plot_timeseries_signal(axss[:4, 1], signal)
        axss[0, 1].set_title("Observed signal")
        plot_timeseries_signal(axss[4:, 0], signal_preprocessed)
        axss[4, 0].set_title("PCA of observed signal")
        plot_timeseries_signal(axss[4:, 1], source)
        axss[4, 1].set_title("Recovered source")

        # Formatting
        for ax in axss.flat:
            ax.tick_params(direction="in")
            ax.set_ylim(-2.5, 2.5)

        # Title
        fig.suptitle(f"Iteration {int(iteration)}")

        # Save
        img_path = f"save/timeseries/optimization/{int(iteration)}.png"
        util.save_fig(
            fig, img_path, dpi=100, tight_layout_kwargs={"rect": [0, 0, 1, 1.0]},
        )
        img_paths.append(img_path)

    # --Make gif
    util.make_gif(img_paths, "save/timeseries/optimization.gif", 10)


if __name__ == "__main__":
    main()

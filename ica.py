import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
import numpyro
import math
import util
import numpy as np


def get_signal(mixing_matrix, source):
    """Compute single signal from a single source
    Args
        mixing_matrix [signal_dim, source_dim]
        source [source_dim]
    
    Returns
        signal [signal_dim]
    """
    return jnp.dot(mixing_matrix, source)


def get_subgaussian_log_prob(source):
    """Subgaussian log probability of a single source.

    Args
        source [source_dim]

    Returns []
    """
    return jnp.sum(jnp.sqrt(jnp.abs(source)))


def get_supergaussian_log_prob(source):
    """Supergaussian log probability of a single source.
    log cosh(x) = log ( (exp(x) + exp(-x)) / 2 )
                = log (exp(x) + exp(-x)) - log(2)
                = logaddexp(x, -x) - log(2)
                   
    https://en.wikipedia.org/wiki/Hyperbolic_functions#Exponential_definitions
    https://en.wikipedia.org/wiki/FastICA#Single_component_extraction

    Args
        source [source_dim]

    Returns []
    """
    return jnp.sum(jnp.logaddexp(source, -source) - math.log(2))


def get_antisymmetric_matrix(raw_antisymmetric_matrix):
    """Returns an antisymmetric matrix
    https://en.wikipedia.org/wiki/Skew-symmetric_matrix

    Args
        raw_antisymmetric_matrix [dim * (dim - 1) / 2]: elements in the upper triangular
            (excluding the diagonal)

    Returns [dim, dim]
    """
    dim = math.ceil(math.sqrt(raw_antisymmetric_matrix.shape[0] * 2))
    zeros = jnp.zeros((dim, dim))
    indices = jnp.triu_indices(dim, k=1)
    upper_triangular = zeros.at[indices].set(raw_antisymmetric_matrix)
    return upper_triangular - upper_triangular.T


def get_orthonormal_matrix(raw_orthonormal_matrix):
    """Returns an orthonormal matrix
    https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map

    Args
        raw_orthonormal_matrix [dim * (dim - 1) / 2]

    Returns [dim, dim]
    """
    antisymmetric_matrix = get_antisymmetric_matrix(raw_orthonormal_matrix)
    dim = antisymmetric_matrix.shape[0]
    eye = jnp.eye(dim)
    return jnp.matmul(eye - antisymmetric_matrix, jnp.linalg.inv(eye + antisymmetric_matrix))


def get_source(signal, raw_mixing_matrix):
    """Get source from signal
    
    Args
        signal [signal_dim]
        raw_mixing_matrix [dim * (dim - 1) / 2]
    
    Returns []
    """
    return jnp.matmul(get_mixing_matrix(raw_mixing_matrix).T, signal)


def get_log_likelihood(signal, raw_mixing_matrix, get_source_log_prob):
    """Log likelihood of a single signal log p(x_n)
    
    Args
        signal [signal_dim]
        raw_mixing_matrix [dim * (dim - 1) / 2]
        get_source_log_prob [source_dim] -> []
    
    Returns []
    """
    return get_source_log_prob(get_source(signal, raw_mixing_matrix))


def get_mixing_matrix(raw_mixing_matrix):
    """Get mixing matrix from a vector of raw values (to be optimized)

    Args
        raw_orthonormal_matrix [dim * (dim - 1) / 2]

    Returns [dim, dim]
    """
    return get_orthonormal_matrix(raw_mixing_matrix)


def get_total_log_likelihood(signals, raw_mixing_matrix, get_source_log_prob):
    """Log likelihood of all signals ∑_n log p(x_n)
    
    Args
        signals [num_samples, signal_dim]
        raw_mixing_matrix [dim * (dim - 1) / 2]
        get_source_log_prob [source_dim] -> []
    
    Returns []
    """
    log_likelihoods = jax.vmap(get_log_likelihood, (0, None, None), 0)(
        signals, raw_mixing_matrix, get_source_log_prob
    )
    return jnp.sum(log_likelihoods)


def update_raw_mixing_matrix(raw_mixing_matrix, signals, get_source_log_prob, lr=1e-3):
    """Update raw mixing matrix by stepping the gradient

    Args:
        raw_mixing_matrix [signal_dim, source_dim]
        signals [num_samples, signal_dim]
        get_source_log_prob [source_dim] -> []
        lr (float)

    Returns
        total_log_likelihood []
        updated_raw_mixing_matrix [signal_dim, source_dim]
    """
    total_log_likelihood, g = jax.value_and_grad(get_total_log_likelihood, 1)(
        signals, raw_mixing_matrix, get_source_log_prob
    )
    return total_log_likelihood, raw_mixing_matrix + lr * g


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
    signal = jax.vmap(get_signal, (None, 0), 0)(mixing_matrix, true_source)

    return true_source, mixing_matrix, signal


def preprocess_signal(signal):
    """Center and whiten the signal
    x_preprocessed = A @ (x - mean)

    Args
        signal [num_samples, signal_dim]
    
    Returns
        signal_preprocessed [num_samples, signal_dim]
        preprocessing_params
            A [signal_dim, signal_dim]
            mean [signal_dim]
    """
    mean = jnp.mean(signal, axis=0)
    signal_centered = signal - jnp.mean(signal, axis=0)

    signal_cov = jnp.mean(jax.vmap(jnp.outer, (0, 0), 0)(signal_centered, signal_centered), axis=0)
    eigenvalues, eigenvectors = jnp.linalg.eigh(signal_cov)
    A = jnp.diag(eigenvalues ** (-1 / 2)) @ eigenvectors.T

    return jax.vmap(jnp.matmul, (None, 0), 0)(A, signal_centered), (A, mean)


def ica(key, signal, get_source_log_prob, num_iterations=1000, lr=1e-3):
    dim = signal.shape[1]

    # Preprocess
    signal_preprocessed, preprocessing_params = preprocess_signal(signal)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(signal_preprocessed[:, 0], signal_preprocessed[:, 1])
    util.save_fig(fig, "save/signal_preprocessed.png")

    # Optim
    raw_mixing_matrix = jax.random.normal(key, (int(dim * (dim - 1) / 2),))

    total_log_likelihoods = []
    raw_mixing_matrices = [raw_mixing_matrix]
    for _ in tqdm.tqdm(range(num_iterations)):
        total_log_likelihood, raw_mixing_matrix = update_raw_mixing_matrix(
            raw_mixing_matrix, signal_preprocessed, get_source_log_prob, lr
        )
        total_log_likelihoods.append(total_log_likelihood.item())
        raw_mixing_matrices.append(raw_mixing_matrix)

    return total_log_likelihoods, raw_mixing_matrices, preprocessing_params


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
    total_log_likelihoods, raw_mixing_matrices, preprocessing_params = ica(
        key, signal, get_subgaussian_log_prob, num_iterations=num_iterations, lr=lr
    )

    # Plot likelihoods
    fig, ax = plt.subplots(1, 1)
    ax.plot(total_log_likelihoods)
    util.save_fig(fig, "save/total_log_likelihoods.png")

    # Print mixing matrix
    signal_preprocessed, (A, mean) = preprocess_signal(signal)
    print(f"Mixing matrix: {jnp.linalg.inv(A) @ get_mixing_matrix(raw_mixing_matrices[-1])}")
    print(f"True mixing matrix: {mixing_matrix}")

    # Plot training progress
    # --Plot pngs
    scatter_kwargs = {"s": 0.5, "c": "black", "marker": "o"}
    img_paths = []
    for iteration in np.linspace(0, num_iterations, 11):
        source = jax.vmap(get_source, (0, None), 0)(
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
        img_path = f"save/ica/{int(iteration)}.png"
        util.save_fig(
            fig, img_path, dpi=400, tight_layout_kwargs={"rect": [0, 0, 1, 1.02]},
        )
        img_paths.append(img_path)

    # --Make gif
    util.make_gif(img_paths, "save/ica.gif", 10)


if __name__ == "__main__":
    main()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
import numpyro
import math
import util


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
    log cosh(u) = log ( (exp(x) + exp(-x)) / 2 )
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


def main():
    key = jax.random.PRNGKey(0)
    num_samples = 1000
    source_dim = 2
    signal_dim = 2

    # Source
    low, high = jnp.full((source_dim,), -math.sqrt(3)), jnp.full((source_dim,), math.sqrt(3))
    true_source_dist = numpyro.distributions.Independent(
        numpyro.distributions.Uniform(low, high), reinterpreted_batch_ndims=1
    )
    key, subkey = jax.random.split(key)
    true_source = true_source_dist.sample(subkey, (num_samples,))

    fig, ax = plt.subplots(1, 1)
    ax.scatter(true_source[:, 0], true_source[:, 1])
    util.save_fig(fig, "save/true_source.png")

    # Mixing matrix
    mixing_matrix = jnp.array([[2, 3], [2, 1]])

    # Observed signal
    signal = jax.vmap(get_signal, (None, 0), 0)(mixing_matrix, true_source)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(signal[:, 0], signal[:, 1])
    util.save_fig(fig, "save/signal.png")

    # Preprocess
    signal_centered = signal - jnp.mean(signal, axis=0)

    signal_cov = jnp.mean(jax.vmap(jnp.outer, (0, 0), 0)(signal_centered, signal_centered), axis=0)
    eigenvalues, eigenvectors = jnp.linalg.eigh(signal_cov)
    signal_whitened = jax.vmap(jnp.matmul, (None, 0), 0)(
        jnp.diag(eigenvalues ** (-1 / 2)) @ eigenvectors.T, signal_centered
    )

    fig, ax = plt.subplots(1, 1)
    ax.scatter(signal_whitened[:, 0], signal_whitened[:, 1])
    util.save_fig(fig, "save/signal_whitened.png")

    # Optim
    key, subkey = jax.random.split(key)
    dim = source_dim
    raw_mixing_matrix = jax.random.normal(subkey, (int(dim * (dim - 1) / 2),))

    num_iterations = 100
    total_log_likelihoods = []
    raw_mixing_matrices = [raw_mixing_matrix]
    for _ in tqdm.tqdm(range(num_iterations)):
        total_log_likelihood, raw_mixing_matrix = update_raw_mixing_matrix(
            raw_mixing_matrix, signal_whitened, get_subgaussian_log_prob
        )
        total_log_likelihoods.append(total_log_likelihood.item())
        raw_mixing_matrices.append(raw_mixing_matrix)

    fig, ax = plt.subplots(1, 1)
    ax.plot(total_log_likelihoods)
    util.save_fig(fig, "save/total_log_likelihoods.png")

    source = jax.vmap(get_source, (0, None), 0)(signal_whitened, raw_mixing_matrix)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(source[:, 0], source[:, 1])
    util.save_fig(fig, "save/recovered_source.png")
    print(f"Mixing matrix: {get_mixing_matrix(raw_mixing_matrix)}")


if __name__ == "__main__":
    main()
import jax
import jax.numpy as jnp
import tqdm
import math


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
    """Gradient-descent based maximum likelihood estimation of the independent component analysis
    (ICA) model

    Args
        key (Jax's PRNG key)
        signal [num_samples, signal_dim]
        get_source_log_prob [source_dim] -> []
        num_iterations (int)
        lr (float)
    
    Returns
        total_log_likelihoods: list of length num_iterations
        raw_mixing_matrices: list of length (num_iterations + 1)
        preprocessing_params
            A [signal_dim, signal_dim]
            mean [signal_dim]

            where the preprocessed signal is obtained by

            matmul(A, (signal - mean))
    """
    dim = signal.shape[1]

    # Preprocess
    signal_preprocessed, preprocessing_params = preprocess_signal(signal)

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

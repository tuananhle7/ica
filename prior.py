import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import math
import util
import tqdm


def get_rotation_matrix(angle):
    cos = math.cos(angle)
    sin = math.sin(angle)
    return jnp.array([[cos, -sin], [sin, cos]])


def get_likelihoods(dist, samples, angles):
    """Compute the average of log prob under `dist` of `samples` rotated by `angles`"""
    log_probs = []
    for angle in angles:
        log_probs.append(
            dist.log_prob(
                jax.vmap(jnp.matmul, (None, 0), 0)(get_rotation_matrix(angle), samples)
            ).mean()
        )
    return jnp.stack(log_probs)


def get_prob(dist, angle, xy):
    """Return the probability under `dist` when `xy` is rotated by `angle`

    Args
        dist: batch_shape [], event_shape [2]
        angle []
        xy: [*shape, 2]

    Returns [*shape]
    """
    return jnp.exp(dist.log_prob(jnp.einsum("ij,...j->...i", get_rotation_matrix(angle), xy)))


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    num_angles = 100
    num_points = 1000
    min_point, max_point = -7, 7
    x, y = jnp.meshgrid(
        jnp.linspace(min_point, max_point, num_points),
        jnp.linspace(min_point, max_point, num_points),
    )
    xy = jnp.stack([x, y], axis=-1)

    angles = jnp.linspace(0, 2 * math.pi, num_angles)
    num_samples = 10000

    # Create dists
    loc = jnp.zeros((2,))
    scale = jnp.ones((2,))
    laplace_dist = numpyro.distributions.Independent(
        numpyro.distributions.Laplace(loc, scale), reinterpreted_batch_ndims=1
    )

    normal_dist = numpyro.distributions.Independent(
        numpyro.distributions.Normal(loc, scale), reinterpreted_batch_ndims=1
    )
    key, subkey = jax.random.split(key)
    normal_samples = normal_dist.sample(subkey, (num_samples,))
    neg_normal_log_probs = -get_likelihoods(laplace_dist, normal_samples, angles)

    key, subkey = jax.random.split(key)
    laplace_samples = laplace_dist.sample(subkey, (num_samples,))
    neg_laplace_log_probs = -get_likelihoods(laplace_dist, laplace_samples, angles)

    studentt_dist = numpyro.distributions.Independent(
        numpyro.distributions.StudentT(2, loc, scale), reinterpreted_batch_ndims=1
    )
    key, subkey = jax.random.split(key)
    studentt_samples = studentt_dist.sample(subkey, (num_samples,))
    neg_studentt_log_probs = -get_likelihoods(laplace_dist, studentt_samples, angles)

    laplace2_dist = numpyro.distributions.Independent(
        numpyro.distributions.Laplace(loc, 2 * scale), reinterpreted_batch_ndims=1
    )
    key, subkey = jax.random.split(key)
    laplace2_samples = laplace2_dist.sample(subkey, (num_samples,))
    neg_laplace2_log_probs = -get_likelihoods(laplace_dist, laplace2_samples, angles)

    # Plotting
    extent = [min_point, max_point, min_point, max_point]
    vmin = 0.0
    vmax = 0.15
    paths = []

    for angle_id, angle in tqdm.tqdm(enumerate(angles)):
        fig = plt.figure(constrained_layout=False, figsize=(6, 6))
        gs = fig.add_gridspec(5, 5)
        ax_main = fig.add_subplot(gs[:, :-1])
        axs = []
        for i in range(5):
            axs.append(fig.add_subplot(gs[i, -1]))

        # MAIN
        ax = ax_main

        ax.plot(angles, neg_laplace2_log_probs, label="laplace2", color="tab:orange")
        ax.scatter(angles[angle_id], neg_laplace2_log_probs[angle_id], color="tab:orange")

        ax.plot(angles, neg_studentt_log_probs, label="studentt", color="tab:red")
        ax.scatter(angles[angle_id], neg_studentt_log_probs[angle_id], color="tab:red")

        ax.plot(angles, neg_laplace_log_probs, label="laplace", color="tab:green")
        ax.scatter(angles[angle_id], neg_laplace_log_probs[angle_id], color="tab:green")

        ax.plot(angles, neg_normal_log_probs, label="gaussian", color="tab:blue")
        ax.scatter(angles[angle_id], neg_normal_log_probs[angle_id], color="tab:blue")

        ax.set_title("Cross-entropy between\nthe rotated true distribution and the prior")
        ax.set_ylabel(f"$E_{{p_{{TRUE}}(z | \\theta)}}[-\log p_z(z)]$")
        ax.set_xlabel(f"Rotation of true distribution $\\theta$")
        ax.set_yticks([])
        ax.set_xticks([i * math.pi / 4 for i in range(9)])
        ax.grid(True, axis="x")
        ax.set_xticklabels([f"${i}\pi / 4$" for i in range(9)])
        ax.tick_params(direction="in")
        ax.legend(title=f"$p_{{TRUE}}(z | \\theta)$")

        # SIDE
        ax = axs[0]
        prob = jnp.exp(laplace_dist.log_prob(xy))
        ax.imshow(prob, extent=extent, cmap="Greys", vmin=vmin, vmax=vmax)
        ax.set_title(f"$p_z(z)$")

        ax = axs[-4]
        prob = get_prob(laplace2_dist, angle, xy)
        ax.imshow(prob, extent=extent, cmap="Oranges", vmin=vmin, vmax=vmax)
        ax.set_title("laplace2")

        ax = axs[-3]
        prob = get_prob(studentt_dist, angle, xy)
        ax.imshow(prob, extent=extent, cmap="Reds", vmin=vmin, vmax=vmax)
        ax.set_title("studentt")

        ax = axs[-2]
        prob = get_prob(laplace_dist, angle, xy)
        ax.imshow(prob, extent=extent, cmap="Greens", vmin=vmin, vmax=vmax)
        ax.set_title("laplace")

        ax = axs[-1]
        prob = get_prob(normal_dist, angle, xy)
        ax.imshow(prob, extent=extent, cmap="Blues", vmin=vmin, vmax=vmax)
        ax.set_title("gaussian")

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        path = f"save/prior/pngs/{angle_id}.png"
        util.save_fig(fig, path, dpi=100)
        paths.append(path)

    util.make_gif(paths, "save/prior/prior.gif", 12)

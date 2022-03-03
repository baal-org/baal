from typing import List

import matplotlib.pyplot as plt
import numpy as np

BG_COLOR = "lavender"
FG_COLORS = [
    "b",
    "g",
    "r",
    "c",
    "m",
    "y",
    "tab:orange",
    "tab:purple",
    "limegreen",
    "yellow",
    "tab:brown",
]


def make_animation_from_data(
    features: np.ndarray, labels: np.ndarray, labelled_at: np.ndarray, classes: List[str]
) -> List[np.ndarray]:
    """
    Make an animation that show the progress of labelling.

    Args:
        features: 2d features representation of the inputs. Shape [samples, 2]
        labels: Label id for each inputs. Shape [samples]
        labelled_at: Index at which the input was labelled. Shape [samples]
        classes: List of classes.

    Returns:
        Animated frames of the labelling process.
        You can then save it locally with:
            `imageio.mimsave('output.gif', frames, fps=3)`
    """
    assert features.ndim == 2 and features.shape[-1] == 2, "Can only plot 2d points!"
    frames = []
    for frame_id in reversed(range(np.max(labelled_at))):
        # New frame
        fig, ax = plt.subplots(figsize=(10, 10))
        # Filter stuff
        currently_labelled = labelled_at > frame_id
        unlabelled_features = features[~currently_labelled]
        labelled_features = features[currently_labelled]
        labelled_labels = labels[currently_labelled]
        unique_labels = np.unique(labelled_labels)

        ax.scatter(
            unlabelled_features[:, 0],
            unlabelled_features[:, 1],
            c=BG_COLOR,
            label="Unlabelled",
            marker="x",
            zorder=2,
        )
        for color, label_name, label_id in zip(FG_COLORS, classes, unique_labels):
            label_mask = labelled_labels == label_id
            pts = labelled_features[label_mask]
            ax.scatter(pts[:, 0], pts[:, 1], c=color, label=label_name, marker="x", zorder=2)
        ax.set_title(
            "{} : {}/{}".format(
                "Labelling progress", currently_labelled.sum(), len(currently_labelled)
            )
        )
        ax.legend(loc="best", ncol=1, prop={"size": 15}, markerscale=3, fancybox=True, shadow=True)
        fig.set_size_inches(15, 10.0)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)
    return frames


if __name__ == "__main__":  # pragma: no cover
    from sklearn.datasets import make_classification
    import imageio

    # 2D input to mimic a t-SNE-like shape.
    X, y = make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1,
        n_classes=3,
    )
    labelled_at = np.random.randint(0, 100, size=[X.shape[0]])
    class_name = ["cow", "dog", "cat"]
    frames = make_animation_from_data(X, y, labelled_at, class_name)
    imageio.mimsave("output.gif", frames, fps=3)

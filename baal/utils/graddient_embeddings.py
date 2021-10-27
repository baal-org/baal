from typing import List, Any
import numpy as np


def find_last_embedding_layer(model: Any, layer_name: str) -> Any:
    """
    Find the embedding layer given the name if the layer.
    Args:
        model: Model
        layer_name: Layer Name
    Returns:
        Module of the embedding layer.
    """
    searchable_str = layer_name.lower().strip()

    for name, layer in model.named_modules():
        if name == searchable_str:
            return layer

    raise ValueError(f"{searchable_str} layer not found in model")


def register_embedding_list_hook(
    model: Any, embeddings_list: List[np.ndarray], layer_name: str
) -> Any:
    def forward_hook(module, inputs, output):
        embeddings_list.append(output)

    embedding_layer = find_last_embedding_layer(model, layer_name)
    print(embedding_layer)
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle


def register_embedding_gradient_hooks(
    model: Any, embeddings_gradients: List[np.ndarray], layer_name: str
) -> Any:
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out)

    embedding_layer = find_last_embedding_layer(model, layer_name)
    hook = embedding_layer.register_full_backward_hook(hook_layers)
    return hook
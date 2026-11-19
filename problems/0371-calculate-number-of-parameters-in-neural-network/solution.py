def calculate_parameters(layers: list[dict]) -> int:
    """
    Calculate the total number of trainable parameters in a neural network.

    Args:
        layers: List of dictionaries, each describing a layer.

    Returns:
        Total number of trainable parameters (int).
    """
    # Your code here
    res = 0
    for l in layers:
        layer_type = l["type"]
        has_bias = l.get("bias", True)

        if layer_type == "dense":
            in_features = l["input_size"]
            out_features = l["output_size"]
            weights = in_features * out_features
            biases = out_features if has_bias else 0
            res += (weights + biases)
        
        if layer_type == "conv2d":
            in_channels = l["in_channels"]
            out_channels = l["out_channels"]
            kernel_size = l["kernel_size"]
            weights = in_channels * out_channels * kernel_size ** 2
            biases = out_channels if has_bias else 0
            res += (weights + biases)
        
    return res
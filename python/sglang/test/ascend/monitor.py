import logging
import time


def create_attention_monitor_factory(config):
    # hook factory
    layer_index = config.get("layer_index", 0)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    def attention_monitor_hook(module, inputs, output):
        # The actual hook function is called during the forward propagation of the self-attention layer.
        timestamp = time.time()

        hidden_states = inputs[1] if inputs else None

        monitor_record = {
            "timestamp": timestamp,
            "layer_index": layer_index,
            "module_type": type(module).__name__,
            "inputs": hidden_states.sum(-1)[:5] if hidden_states is not None else None,
            "outputs": output.sum(-1)[:5],
        }

        logging.info(f"hook effect: {monitor_record}")

        return output

    return attention_monitor_hook

# hook
import logging
import sys
import time

def create_attention_monitor_factory(config):
    """
    钩子工厂函数
    config: from --forward hooks
    """
    layer_index = config.get("layer_index", 0)
    log_file="/data/y30082119/hook.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",  # 添加时间戳和日志级别
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file,mode="a"),
            logging.StreamHandler()
        ]
    )
    def attention_monitor_hook(module, inputs, output):
        """
        实际钩子函数,在self-attention层的前向传播时被调用
        """

        # 获取时间戳
        timestamp = time.time()

        # 提取输入信息
        hidden_states = inputs[1] if inputs else None

        # 记录关键信息
        monitor_record = {
            "timestamp": timestamp,
            "layer_index": layer_index,
            "module_type": type(module).__name__,
            "inputs": hidden_states.sum(-1)[:5] if hidden_states is not None else None,
            "outputs": output.sum(-1)[:5].tolist(),
        }
        # 实时打印监控信息

        print(f"[AttentionMonitor] Layer {layer_index} - "
              f"Input: {monitor_record['inputs']},"
              f"Output: {output.sum(-1)[:5]},")

        logging.info(f"hook effect: {monitor_record}")

        # 必须返回输出，否则会中断前向传播
        return output

    return attention_monitor_hook

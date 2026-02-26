import logging
import time


def create_attention_monitor_factory(config):
    """
    钩子工厂函数
    config: from --forward hooks
    """
    layer_index = config.get("layer_index", 0)
    log_file = "hook.log"

    # ========== 仅修改这部分日志配置 ==========
    # 1. 检查日志是否已配置，避免重复调用basicConfig失效
    logger = logging.getLogger()
    if not logger.handlers:
        # 2. 新增encoding=utf-8解决中文乱码
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            encoding="utf-8"  # 新增：解决中文乱码
        )

    # =========================================

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
            "outputs": output.sum(-1)[:5],
        }
        # 实时打印监控信息

        print(f"[AttentionMonitor] Layer {layer_index} - "
              f"Input: {monitor_record['inputs']},"
              f"Output: {output.sum(-1)[:5]},")

        logging.info(f"hook effect: {monitor_record}")

        # 必须返回输出，否则会中断前向传播
        return output

    return attention_monitor_hook

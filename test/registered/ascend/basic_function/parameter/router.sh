unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export HCCL_SOCKET_IFNAME="enp23s0f3"
export GLOO_SOCKET_IFNAME="enp23s0f3"
 python -m sglang_router.launch_router \
--decode http://192.168.0.244:30000 \
--prefill http://192.168.0.188:30000 \
--pd-disaggregation \
--mini-lb \
--policy cache_aware \
--host 127.0.0.1 \
--port 6688

# Simplified benchmarking script from cleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/dqn.sh

OMP_NUM_THREADS=1 python -m src.benchmark \
    --env-ids PhoenixNoFrameskip-v4 \
    --command "python injection_dqn.py --track --apply-injection --total-timesteps=15000000 --injection_step=4000000" \
    --num-seeds 3 \
    --workers 3
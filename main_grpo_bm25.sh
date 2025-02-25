set -x
MODEL_PATH_BASE=/root/autodl-tmp/pretrained_models/
MODEL_NAME=Qwen2.5-7B-Instruct
MODEL_PATH=$MODEL_PATH_BASE$MODEL_NAME
TRAIN_DATA=data/bm25_eval/full/train.parquet
TEST_DATA=data/bm25_eval/full/test.parquet
#export CUDA_VISIBLE_DEVICES=3,4,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$TEST_DATA \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=8000 \
    data.max_response_length=500 \
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='GRPO_BRIGHT_full_bm25_autodl' \
    trainer.experiment_name=0212_bugfix_sampledoc_$MODEL_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=save_models \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@ 2>&1 | tee grpo.log

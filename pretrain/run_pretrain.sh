EMB_NAME=entity_emb
LR=1e-4

python large_emb.py --lr $LR --total_client 8 --emb_name $EMB_NAME \
  --ent_emb ../wikidata5m_alias_emb/entities.npy &
python -m torch.distributed.launch --nproc_per_node=8 run_pretrain.py --name CoLAKE --data_prop 1.0 \
  --batch_size 2048 --lr $LR --ent_lr $LR --epoch 1 --grad_accumulation 16 --save_model --emb_name $EMB_NAME \
  --n_negs 200 --beta 0.98
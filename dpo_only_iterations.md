# DPO-only Iterations (10-step run)

| Iter | DPO config change | Train run | Eval run | BLEU | Keep |
|---:|---|---|---|---:|---|
| 1 | baseline (`lr=5e-6,beta=0.1,max_steps=60,n_train=4096`) | https://modal.com/apps/lothnic/main/ap-HYvLhpM5xX3hFjRCpOvVFu | https://modal.com/apps/lothnic/main/ap-DBUOVNI03L2D7RL8oCYsp2 | 5.2573 | yes |
| 2 | `learning_rate=1e-5` | https://modal.com/apps/lothnic/main/ap-LJJY6Z3hhHm9urvkRqNFal | https://modal.com/apps/lothnic/main/ap-gASwEcuHbRA0YpF63ytlVv | 5.2894 | yes |
| 3 | `dpo_beta=0.2` | https://modal.com/apps/lothnic/main/ap-1VICWdUCzZ0GQhPcPs2I8u | https://modal.com/apps/lothnic/main/ap-c0nSGA3rvP9x8VD6yxX2eY | 5.3246 | yes |
| 4 | `max_steps=90` | https://modal.com/apps/lothnic/main/ap-kfr5EUfPzrnPPXIamxlgRU | https://modal.com/apps/lothnic/main/ap-zGkpb0LbEO0AizwjShEUok | 5.3847 | yes |
| 5 | `n_train=8192` | https://modal.com/apps/lothnic/main/ap-bldcHSq0MZutzQc8CIlJBo | https://modal.com/apps/lothnic/main/ap-GVYV1YyfcO0G1RvoOO94we | 5.2728 | no |

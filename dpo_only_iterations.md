# DPO-only Iterations (10-step run)

| Iter | DPO config change | Train run | Eval run | BLEU | Keep |
|---:|---|---|---|---:|---|
| 1 | baseline (`lr=5e-6,beta=0.1,max_steps=60,n_train=4096`) | https://modal.com/apps/lothnic/main/ap-HYvLhpM5xX3hFjRCpOvVFu | https://modal.com/apps/lothnic/main/ap-DBUOVNI03L2D7RL8oCYsp2 | 5.2573 | yes |
| 2 | `learning_rate=1e-5` | https://modal.com/apps/lothnic/main/ap-LJJY6Z3hhHm9urvkRqNFal | https://modal.com/apps/lothnic/main/ap-gASwEcuHbRA0YpF63ytlVv | 5.2894 | yes |
| 3 | `dpo_beta=0.2` | https://modal.com/apps/lothnic/main/ap-1VICWdUCzZ0GQhPcPs2I8u | https://modal.com/apps/lothnic/main/ap-c0nSGA3rvP9x8VD6yxX2eY | 5.3246 | yes |
| 4 | `max_steps=90` | https://modal.com/apps/lothnic/main/ap-kfr5EUfPzrnPPXIamxlgRU | https://modal.com/apps/lothnic/main/ap-zGkpb0LbEO0AizwjShEUok | 5.3847 | yes |
| 5 | `n_train=8192` | https://modal.com/apps/lothnic/main/ap-bldcHSq0MZutzQc8CIlJBo | https://modal.com/apps/lothnic/main/ap-GVYV1YyfcO0G1RvoOO94we | 5.2728 | no |
| 6 | `dpo_beta=0.3` (n_train back 4096) | https://modal.com/apps/lothnic/main/ap-OZCq50N1cyuAgePpIYBPFR | https://modal.com/apps/lothnic/main/ap-zNGlYDher5zGYMNS8FQaHr | 5.2628 | no |
| 7 | `batch_size=4, grad_accum=4` (same effective batch) | https://modal.com/apps/lothnic/main/ap-wOWlGVSvN1JB0OMAisqRub | https://modal.com/apps/lothnic/main/ap-EMZ6HR3pbcsXYLkaXMLl5n | 5.3446 | no |
| 8 | `learning_rate=8e-6` | https://modal.com/apps/lothnic/main/ap-X6tj1UzY0XcTDwWaB2gU1i | https://modal.com/apps/lothnic/main/ap-0jXDdGyzE75hdwcbvjHOih | 5.3646 | no |
| 9 | longer train truncation (`max_source_length=80,max_length=160`) | https://modal.com/apps/lothnic/main/ap-CmeM2gwlx8xvkcsJuHh7T1 | https://modal.com/apps/lothnic/main/ap-jPnqokwO2ftG16uVxCuQP1 | 5.4359 | yes |

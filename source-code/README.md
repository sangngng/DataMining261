
## Fine-tuned encoders (blend_ft)
`src/data_prep/finetune_encoder.py` (run on a Colab T4: `python finetune_encoder.py BAAI/bge-base-en-v1.5 bge`, and `intfloat/e5-base-v2 e5`) fine-tunes a pretrained encoder (BGE: Xiao et al. 2023; E5: Wang et al. 2022) on the train labels only, using text `VENUE YEAR: title`, four sigmoid heads for P(y>=k), k=2..5, 5 epochs, same CV folds; test predictions are the mean of 3 seeds trained on all train (`data/ft_{bge,e5}_test.npy`, OOF in `data/ft_*_oof.npy`).
`python -m src.make_submission --model blend_ft --keep 0.65` = equal five-part rank average (stack, MiniLM-title, BGE-title, fine-tuned BGE, fine-tuned E5).
CV (2 reps, keep 0.65): blend 0.6175 -> blend_ft 0.6223 (AUC 0.808 -> 0.823); gain +0.005 is small and not statistically significant (95% CI about [-0.007, +0.017]).
Confirmation on 3 fresh CV repetitions (reps 2-4; fine-tune script variant: `src/data_prep/finetune_encoder.py` with `REPS=(2,3,4)`): over all 5 reps, blend 0.6130 -> blend_ft 0.6196 (+0.0066, 95% CI about [-0.006, +0.020]); AUC(y>1) 0.807 -> 0.823, higher in 5 of 5 reps.

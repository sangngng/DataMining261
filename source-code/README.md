
## Fine-tuned encoders (blend_ft)
`src/data_prep/finetune_encoder.py` (run on a Colab T4: `python finetune_encoder.py BAAI/bge-base-en-v1.5 bge`, and `intfloat/e5-base-v2 e5`) fine-tunes a pretrained encoder (BGE: Xiao et al. 2023; E5: Wang et al. 2022) on the train labels only, using text `VENUE YEAR: title`, four sigmoid heads for P(y>=k), k=2..5, 5 epochs, same CV folds; test predictions are the mean of 3 seeds trained on all train (`data/ft_{bge,e5}_test.npy`, OOF in `data/ft_*_oof.npy`).
`python -m src.make_submission --model blend_ft --keep 0.65` = equal five-part rank average (stack, MiniLM-title, BGE-title, fine-tuned BGE, fine-tuned E5).
CV (2 reps, keep 0.65): blend 0.6175 -> blend_ft 0.6223 (AUC 0.808 -> 0.823); gain +0.005 is small and not statistically significant (95% CI about [-0.007, +0.017]).
Confirmation on 3 fresh CV repetitions (reps 2-4; fine-tune script variant: `src/data_prep/finetune_encoder.py` with `REPS=(2,3,4)`): over all 5 reps, blend 0.6130 -> blend_ft 0.6196 (+0.0066, 95% CI about [-0.006, +0.020]); AUC(y>1) 0.807 -> 0.823, higher in 5 of 5 reps.

### Four fine-tuned encoders (blend_ft4)
`src/data_prep/finetune_encoder_cv5.py MODEL TAG` (Colab T4; needs `cv_pack5.csv` = ids, labels, venue, year, title, fold0-4 and the current blend's OOF utilities cur0-4) fine-tunes on all 5 CV repetitions and the full train set. Models: BGE-base (BAAI/bge-base-en-v1.5), E5-base (intfloat/e5-base-v2), SciBERT (allenai/scibert_scivocab_uncased; Beltagy et al. 2019) and MPNet (sentence-transformers/all-mpnet-base-v2; Song et al. 2020), tags bge, e5, sci, mpnet. Test predictions: `data/ft_{bge,e5,sci,mpnet}_test.npy`.
`python -m src.make_submission --model blend_ft4 --keep 0.65` = 0.5 * mean(stack, MiniLM-title, BGE-title utilities) + 0.5 * mean(four fine-tuned encoders), rank-normalised.
CV over 5 reps (keep 0.65): blend 0.6130 -> + 2 encoders 0.6200 -> + 4 encoders 0.6230 (AUC 0.807 / 0.824 / 0.827); P(better than blend) about 0.93, CI still includes 0.

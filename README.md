# Emotions-and-Literary-Periods
Data and Code of "Emotions and Literary Periods" (submission to DH2022 conference)
## Files

metadata.tsv  - Corpus metadata (Poem Ids, Epochs, Authors, Titels, Publication Dates, Source Anthologies) <br>
finetune_model.py - performs finetuning of one emmtion group <br>
start_finetuning.sh - calls finetune_model.py for all emotion groups with original parameter setting <br>
prediction.ipynb - Notebook to infer predictions from model checkpoints <br>
shaver_predictions.tsv - Combined result of all model inferences <br>
<br>
pretraining/pretrainer.py - code for unsupervised pretraining on poems <br>
pretraining/start_pretraining.sh - calls pretrainer.py with original parameters <br>

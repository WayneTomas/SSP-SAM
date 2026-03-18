dataset='gref_umd'

output_dir='outputs/your_save_folder'

checkpoint_best = True
checkpoint_latest = True
batch_size=8
epochs=100
lr_drop=60
freeze_epochs=20

freeze_modules=['sam.mask_decoder']


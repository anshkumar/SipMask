python3 train.py -tfrecord_train_dir '/home/deploy/ved/coco/records/train' -tfrecord_val_dir '/home/deploy/ved/coco/records/val' -label_map 'label_map.pbtxt' -train_iter '1200000' -save_interval 100 -lr_total_steps '1200000' -img_h '550' -img_w '550' -batch_size '8' -num_class '90' -base_model_trainable True -print_interval 100 -save_interval 50000 -valid_iter 5000 

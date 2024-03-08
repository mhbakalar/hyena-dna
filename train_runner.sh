for i in {0..4}; do
  echo "Training fold $i"
  python -m train wandb=null experiment=mpra/tiny.yaml dataset.kfold=$i
done

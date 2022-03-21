CUDA_VISIBLE_DEVICES='0' python main.py --model resnet32 --save_dir cifar100_resnet32 --config_path ./configs/20220223_cifar100.yml

CUDA_VISIBLE_DEVICES='0' python main.py --model vgg16 --save_dir cifar100_vgg16 --config_path ./configs/20220223_cifar100.yml

CUDA_VISIBLE_DEVICES='0' python main.py --model densenetd40k12 --save_dir cifar100_densenetd40k12 --config_path ./configs/20220223_cifar100.yml

CUDA_VISIBLE_DEVICES='0' python main.py --model resnet110 --save_dir cifar100_resnet110 --config_path ./configs/20220223_cifar100.yml

CUDA_VISIBLE_DEVICES='0' python main.py --model wide_resnet20_8 --save_dir cifar100_wide_resnet20_8 --config_path ./configs/20220223_cifar100.yml

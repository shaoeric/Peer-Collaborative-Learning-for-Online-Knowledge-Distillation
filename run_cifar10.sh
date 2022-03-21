CUDA_VISIBLE_DEVICES='1' python main.py --model resnet32 --save_dir cifar10_resnet32 --config_path ./configs/20220309_cifar10.yml

CUDA_VISIBLE_DEVICES='1' python main.py --model vgg16 --save_dir cifar10_vgg16 --config_path ./configs/20220309_cifar10.yml

CUDA_VISIBLE_DEVICES='1' python main.py --model densenetd40k12 --save_dir cifar10_densenetd40k12 --config_path ./configs/20220309_cifar10.yml

CUDA_VISIBLE_DEVICES='1' python main.py --model resnet110 --save_dir cifar10_resnet110 --config_path ./configs/20220309_cifar10.yml

CUDA_VISIBLE_DEVICES='1' python main.py --model wide_resnet20_8 --save_dir cifar10_wide_resnet20_8 --config_path ./configs/20220309_cifar10.yml

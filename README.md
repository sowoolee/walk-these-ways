# Walk-These-Ways Environment Setting


## initial setting
### conda environment setting
#### pytorch

    pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

#### isaacgym
https://developer.nvidia.com/isaac-gym 에서 다운로드 후

    tar -xf IsaacGym_Preview_4_Package.tar.gz
    
    cd isaacgym/python && pip install -e .
잘 설치되었는지 확인하려면

    cd examples && python 1080_balls_of_solitude.py    
#### go1_gym
clone한 repo에서

    pip install -e .
#### verifying

    cd scripts && python test.py
## training
학습할 때

    cd scripts && python train.py

## saving
데이터를 뽑을 때 

    cd scripts && python save.py

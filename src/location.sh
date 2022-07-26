###
 # @Descripttion: 
 # @version: 
 # @Author: Yinan Mei
 # @Date: 2022-07-26 14:12:07
 # @LastEditors: Yinan Mei
 # @LastEditTime: 2022-07-26 14:13:14
### 

PYTHONPATH=$(pwd)

cecho(){
    RED="\033[0;31m"
    GREEN="\033[0;32m"
    YELLOW="\033[1;33m"
    # ... ADD MORE COLORS
    NC="\033[0m" # No Color

    printf "${!1}${2} ${NC}\n"
}


cecho RED "Location - RLMiner Repair"
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Location --num 2559 --stopreward 0.01 --supp 10 --step-per-epoch 1000 --epoch 5 --training-num 1 --decay 0.0002 --update-per-step 0.2 --maxd 600 
CUDA_VISIBLE_DEVICES=0 python discover.py --dataset Location --num 2559 --stopreward 0.01 --supp 10 --maxd 600 
python eval.py --dataset Location --num 2559 --method RLMiner

cecho RED "Location - EnuMiner Repair"
python enuminer.py --dataset Location --num 2559 --supp 10 --maxd 600
python eval.py --dataset Location --num 2559 --method EnuMiner

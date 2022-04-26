###
 # @Descripttion: 
 # @version: 
 # @Author: Yinan Mei
 # @Date: 2022-04-26 07:37:12
 # @LastEditors: Yinan Mei
 # @LastEditTime: 2022-04-26 07:39:47
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


cecho RED "Covid - RLMiner Repair"
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Covid --num 2500 --stopreward 0.01 --supp 150 --step-per-epoch 1000 --epoch 5 --training-num 1 --decay 0.0002 --update-per-step 0.2 --maxd 200 
CUDA_VISIBLE_DEVICES=0 python discover.py --dataset Covid --num 2500 --stopreward 0.01 --supp 150 --maxd 200
python eval.py --dataset Covid --num 2500 --method RLMiner


cecho RED "Covid - EnuMiner Repair"
python enuminer.py --dataset Covid --num 2500 --supp 150
python eval.py --dataset Covid --num 2500 --method EnuMiner
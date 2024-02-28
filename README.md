# MMFI
A repository of **'[Multi-Scenario and Multi-Task Aware Feature Interaction for Recommendation System](). ACM Transactions on Knowledge Discovery from Data (TKDD), 2024. '**

## Abstract
> Multi-scenario and multi-task recommendation can use various feedback behaviors of users in different scenarios to learn users' preferences and then make recommendations, which has attracted attention. However, the existing work ignores feature interactions and the fact that a pair of feature interactions will have differing levels of importance under different scenario-task pairs, leading to sub-optimal user preference learning. In this paper, we propose a Multi-scenario and Multi-task aware Feature Interaction model, dubbed MMFI, to explicitly model feature interactions and learn the importance of feature interaction pairs in different scenarios and tasks. Specifically, MMFI first incorporates a pairwise feature interaction unit and a scenario-task interaction unit to effectively capture the interaction of feature pairs and scenario-task pairs. Then MMFI designs a scenario-task aware attention layer for learning the importance of feature interactions from coarse-grained to fine-grained, improving the model's performance on various scenario-task pairs. More specifically, this attention layer consists of three modules: a fully shared bottom module, a partially shared middle module, and a specific output module. Finally, MMFI adapts two sparsity-aware functions to remove some useless feature interactions. Extensive experiments on two public datasets demonstrate the superiority of the proposed method over the existing multi-task recommendation, multi-scenario recommendation, and multi-scenario \& multi-task recommendation models.

<center>
<img src="./MMFI.png" alt="MMFI" width="800"/>
</center>


## Citation

If you find our paper or this resource helpful, please consider cite:
```
@article{MMFI_TKDD_2024,
  title={Multi-Scenario and Multi-Task Aware Feature Interaction for Recommendation System},
  author={Song, Derun and Yang, Enneng and Guo, Guibing and Shen, Li and Jiang, Linying and Wang, Xingwei. },
  journal={ACM Transactions on Knowledge Discovery from Data},
  volume={},
  number={},
  pages={},
  year={2024},
  publisher={ACM New York, NY}
}
```
Thanks!

## Requirements
- Python == 3.6.13
- PyTorch == 1.1.5
- numpy == 1.10.2
- datatable == 0.11.1

## Datasets
- [AliExpress Dataset](https://github.com/easezyc/Multitask-Recommendation-Library): This is a dataset gathered from real-world traffic logs of the search system in AliExpress.  
- [Tecrec Dataset](https://github.com/yuangh-x/2022-NIPS-Tenrec): Tenrec is a large-scale multipurpose benchmark dataset for recommender systems where data was collected from two feeds (articles and videos) recommendation platforms.


##  Run
> python main.py

## Acknowledgement
Our implementation references the code below, thanks to them.

[MTReclib](https://github.com/easezyc/Multitask-Recommendation-Library)

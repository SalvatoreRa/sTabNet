# sTabNet

Code for **Escaping the Forest: Sparse Interpretable Neural Networks for Tabular Data**

preprint is available [here](https://arxiv.org/abs/2410.17758)

![general scheme of the paper](https://github.com/SalvatoreRa/sTabNet/blob/main/algorithm.png)

## How to cite

```
@misc{raieli2024escapingforestsparseinterpretable,
      title={Escaping the Forest: Sparse Interpretable Neural Networks for Tabular Data}, 
      author={Salvatore Raieli and Abdulrahman Altahhan and Nathalie Jeanray and Stéphane Gerart and Sebastien Vachenc},
      year={2024},
      eprint={2410.17758},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.17758}, 
}
```

## Dataset used in the work

* [METABRIC](https://www.cbioportal.org/study/summary?id=brca_metabric)
* [TCGA Breast](https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018)
* [TCGA lung](https://www.cbioportal.org/study/summary?id=luad_tcga_gdc)
* [TISCH single cell](https://tisch.comp-genomics.org/)

## Synthetic data

We used [make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) from scikit-learn to create the synthetic data
```python 
from sklearn.datasets import make_classification

n_feat =100
n_inf = 10
n_red =0
n_rep=0
n_classes=6
class_sep = 0.1
X, y = make_classification(
    n_samples=1000,
    n_features=n_feat,
    n_informative=n_inf,
    n_redundant=n_red,
    n_repeated=0,
    n_classes=n_classes,
    n_clusters_per_class=1,
    class_sep= class_sep,
    random_state=0,
    shuffle = False,
)
col_names =['col_' + str(i) for i in range(n_feat) ]
X = pd.DataFrame(X, columns= col_names)

```


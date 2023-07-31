# GAIN

###### <h3> Overview
 
This repository contains the codes of a novel approach, **GAIN**, to aggregating knowledge for learning representations by graph convolutional neural networks. The **GAIN** architecture is developed to address multi-class road type classification problem inspired by [GraphSAGE](http://snap.stanford.edu/graphsage/).
Road network graph datasets are generated from OpenStreeMap (OSMnx) and preprocessed according to the corresponding settings. Representation learning improves by application of a search mechanism in the local and the global neighborhood of a graph node.
Anyone interested in using GAIN architecture, please cite the following [paper](https://arxiv.org/abs/2107.07791):

    @article {gharaee2021pr} {
      author = {Gharaee, Zahra and Kowshik, Shreyas and Stromann, Oliver and Felsberg, Michael},
        title = {Graph representation learning for road type classification},
        booktitle = {Pattern Recognition},
        year = {2021}
        page = {}
        volume = {120}
        DOI = {https://doi.org/10.1016/j.patcog.2021.108174}
      }
    }
   
###### <h3> Requirement
Use the packages mentioned in the requirements.txt file to generate road network graphs and to run experiments.

###### <h3> Road network graph generation
Run roadnetwork_graphs.py scripts in codes folder in order to generate transductive road network graphs of Linköping city and inductive road network graphs of Sweden country extracted from OpenStreetMap (OSMnx). Running roadnetwork_graphs.py also generates id-map, class-map, raw features/attributes and the pairs of topological neighbors. A set of transductive and inductive road network graphs of Linköping city and Sweden country are available in graph_data_GainRepo folder.
 
Image below shows road network graph of the Linköping city representing our transductive data set. Road-type class labels of the original graph are described as following and its line graph representation is overlaid in black:
* Class1: red
* Class2: orange
* Class3: yellow
* Class4: skyblue
* Class5: lime

 
| ![Image of Yaktocat](https://github.com/zahrag/GAIN/blob/main/graph_data_GainRepo/osm_transductive/linkoping-osm.png) | 
|:--:| 
| *Road network graph of Linköping city area.* |

 
###### <h3> Road types classification
run_supervised.py and run_unsupervised.py python scripts contans the main codes and the configurations of hyperparameters to run experiments for supervised and unsupervised settings, respectively. 
 
###### <h3> Results
osm_eval.py python script evaluates the representation vectors generated by GAIN for road type classification. Running this script shows the performance results of applying random-baseline, raw-features, and representation vectors of a pre-trained GAIN model, available in logs_GainRepo folder, to classify road networks of transductive and inductive test datasets in both supervised and unsupervised settings. 
 
###### <h4> Contributors: Zahra Gharaee (zahra.gharaee@liu.se) & Shreyas Kowshik (shreyaskowshik@iitkgp.ac.in) & Oliver Stromann(oliver.stromann@liu.se)

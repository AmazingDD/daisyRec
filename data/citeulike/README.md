Dataset *citeulike-a* for CTRSR
============

This dataset, citeulike-a, was used in the IJCAI paper 'Collaborative Topic Regression with Social Regularization' [Wang, Chen and Li]. It was collected from CiteULike and Google Scholar. CiteULike allows users to create their own collections of articles. There are abstracts, titles, and tags for each article. Other information like authors, groups, posting time, and keywords is not used in this paper 'Collaborative Topic Regression with Social Regularization' [Wang, Chen and Li]. The details can be found at http://www.citeulike.org/faq/data.adp. 

It is partly from [Wang and Blei]. Note that the original dataset in [Wang and Blei] does not contain relations between items. We collect the tag information from CiteULike and citations from Google Scholar.

The text information (item content) of citeulike-a is preprocessed by following the same procedure as that in [Wang and Blei].

Some statistics are listed as follows:

| Entity | Total Number |
| -------|----------|
|users 		|			5551 |
|items 		|			16980 |
|tags 			|		46391 |
|citations |				44709 |
|user-item pairs |		204987 |

DATA FILES
citations.dat	citations between articles
item-tag.dat	tags corresponding to articles, one line corresponds to tags of one article (note that this is the version prior to preprocess thus would have more tags than used in the paper)
mult.dat		bag of words for each article
raw-data.csv	raw data
tags.dat		tags, sorted by tag-id's
users.dat		rating matrix (user-item matrix)
vocabulary.dat	corresponding words for file mult.dat

## Data Format
In citation.dat, each line corresponds to the edges linked to one node. For example, Line 1: 3 2 485 3284 means there are 3 edges linked to node 0 (0-based indexing), their (0-based) ID's are 2, 485, and 3284. In other words article 'The metabolic world of escherichia coli is not small' is linked to 'Exploring complex networks', 'Community structure in social and biological networks', and 'Reconstruction of metabolic networks from genome data and analysis of their global structure for various organisms'.

The file tags.dat lists all the tags. 

The file item-tag.dat tells us what tags each node (article) has. For example, the first line: '17 4276 32443 37837 3378 7650 44590 42810 28819 43806 3805 25497 23779 42140 12234 37386 30698 43503' means that node 0 has 17 tags, and their ID's are 4276, ..., 43503.

raw-data.csv: Note that different from other dat files that use 0-based indexing (ID counts from 0), raw-data.csv uses 1-based indexing (ID counts from 1).


## Reference:
[Collaborative Topic Regression with Social Regularization](http://wanghao.in/paper/IJCAI13_CTRSR.pdf)
```
@inproceedings{DBLP:conf/ijcai/WangCL13,
  author    = {Hao Wang and
               Binyi Chen and
               Wu-Jun Li},
  title     = {Collaborative Topic Regression with Social Regularization
               for Tag Recommendation},
  booktitle = {IJCAI},
  year      = {2013}
}
```

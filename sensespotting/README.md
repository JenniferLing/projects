### SenseSpotting tool

This git directory contains a Python implementation of SenseSpotting based on the initial approach of Carpuat et al. (2013). The code was created within the context of a master thesis.  
DeepSenseSpotting consists of two architecture to address the SenseSpotting task: one MLP-based and one with context embeddings.  
SenseSpotting2 contains the re-implementation of the initial approach (including corpus preprocessing, feature extraction, cross validation and ablation study).  

Master thesis abstract:  
Error analysis of state-of-the-art machine translation systems revealed that translation quality
is highly reduced by the occurrence of unseen words. Especially when moving to a new
domain, word senses change dramatically. However, even sense inventories often miss domain
specific senses and are therefore not able to prevent these translation errors.  
We study the SenseSpotting approach for solving this important problem. SenseSpotting identifies unseen
senses using a logistic regression classifier. We use parallel corpora from two different domains
and spot if a word shifts its sense from one domain to another. Our approach integrates multiple
features, which demonstrated good performances in the fields of word sense disambiguation,
word sense induction and domain adaptation. Due to the combination of these features, we
are able to capture the nature of a word that gained a new, unseen sense. In a first step,
we reproduced the original SenseSpotting approach of Carpuat et al. (2013). Modifying
this approach and replacing the classifier by a deep neural network, we finally achieved a
recall of 88.67% and an F1 score of 81.07%. Thereby, our neural classifier performs a reliable
identification of new domain senses. With SenseSpotting, we therefore provide a tool which can
be integrated into the translation process to improve the performance of machine translation
systems.


Initial approach:  
Carpuat, Marine, et al. "Sensespotting: Never let your parallel data tie you to an old domain." Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Vol. 1. 2013.

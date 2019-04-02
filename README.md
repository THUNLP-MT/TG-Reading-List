# Text Generation Reading List
This is a text generation reading list by the Tsinghua Natural Language Processing Group.

* [Datasets](#datasets)
    * [Story Generation](#story_generation)
    * [Text Generation](#text_generation)
* [Tools](#tools)
* [Papers](#papers)
    * [Prepare Papers](#prepare_papers)
    * [Seq2Seq Based Methods](#seq2seq_based)
    * [Variational Autoencoder Based Methods](#vae_based)
    * [Generative Adversarial Nets Based Methods](#gan_based)
    * [Reinforcement Learning Based Methods](#rl_based)
    * [Knowledge Based Methods](#kbs_based)

<h2 id="datasets">Datasets</h2>

<h3 id="story_generation">Story Generation</h2>

* [ROCStories](http://cs.rochester.edu/nlp/rocstories/): Mostafazadeh, Nasrin and Chambers, Nathanael and He, Xiaodong and Parikh, Devi and Batra, Dhruv and Vanderwende, Lucy and Kohli, Pushmeet and Allen, James. 2016. [A Corpus and Evaluation Framework for Deeper Understanding of Commonsense Stories](https://www.aclweb.org/anthology/N16-1098). In *Proceedings of NAACL 2016*.
* [VIST](http://visionandlanguage.net/VIST/): Huang, Ting-Hao (Kenneth) and Ferraro, Francis and Mostafazadeh, Nasrin and Misra, Ishan and Agrawal, Aishwarya and Devlin, Jacob and Girshick, Ross and He, Xiaodong and Kohli, Pushmeet and Batra, Dhruv and Zitnick, C. Lawrence and Parikh, Devi and Vanderwende, Lucy and Galley, Michel and Mitchell, Margaret. 2016. [Visual Storytelling](https://www.aclweb.org/anthology/N16-1147v1). In *Proceedings of ACL 2016*.
* [WritingPrompts](https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz): Fan, Angela and Lewis, Mike and Dauphin, Yann. 2018. [Hierarchical Neural Story Generation](http://aclweb.org/anthology/P18-1082). In *Proceedings of ACL 2018*.

<h3 id="text_generation">Text Generation</h2>

* [Yelp Review Generation Dataset](https://drive.google.com/open?id=1xCt04xWrVhbrSA7T5feV2WSukjmD4SnK): Xu, Jingjing and Ren, Xuancheng and Lin, Junyang and Sun, Xu. 2018. [Diversity-Promoting GAN: A Cross-Entropy Based Generative Adversarial Network for Diversified Text Generation](http://www.aclweb.org/anthology/D18-1428). In *Proceedings of EMNLP 2018*.
* [Amazon Review Generation Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html): McAuley, Julian John and Leskovec, Jure. 2013. [From Amateurs to Connoisseurs: Modeling The Evolution of User Expertise Through Online Reviews](https://arxiv.org/pdf/1303.4402). In *Proceedings of WWW 2013*.

<h2 id="tools">Tools</h2>

* Hu, Zhiting and Yang, Zichao and Zhao, Tiancheng and Shi, Haoran and He, Junxian and Wang, Di and Ma, Xuezhe and Liu, Zhengzhong and Liang, Xiaodan and Qin, Lianhui and others. 2018. [Texar:  A Modularized, Versatile, and Extensible Toolbox for Text Generation](http://www.aclweb.org/anthology/W18-2503). In *Proceedings of ACL 2018*. ([GitHub](https://github.com/asyml/texar))
* Zhu, Yaoming and Lu, Sidi and Zheng, Lei and Guo, Jiaxian and Zhang, Weinan and Wang, Jun and Yu, Yong. 2018. [Textgen: A Benchmarking Platform for Text Generation Models](https://arxiv.org/pdf/1802.01886). In *Proceedings of SIGIR 2018*. ([GitHub](https://github.com/geek-ai/Texygen))

<h2 id="papers">Papers</h2>

<h3 id="prepare_papers">Prepare Papers</h2>

* Kingma, Diederik P and Welling, Max. 2014. [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114). In *Proceedings of ICLR 2014*. ([Citation](https://scholar.google.com/scholar?cites=10486756931164834716&as_sdt=2005&sciodt=0,5&hl=en): 4,317)
* Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. [Sequence to Sequence Learning
with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf). In *Proceedings of NIPS 2014*. ([Citation](https://scholar.google.com/scholar?cites=13133880703797056141&as_sdt=2005&sciodt=0,5&hl=en): 6,076)
* Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua. 2014. [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf). In *Proceedings of NIPS 2014*. ([Citation](https://scholar.google.com/scholar?cites=11977070277539609369&as_sdt=2005&sciodt=0,5&hl=en): 7,952)
* Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf). In *Proceedings of ICLR 2015*. ([Citation](https://scholar.google.com/scholar?cites=9430221802571417838&as_sdt=2005&sciodt=0,5&hl=en): 6,317)
* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. [Attention is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf). In *Proceedings of NIPS 2017*. ([Citation](https://scholar.google.com/scholar?cites=2960712678066186980&as_sdt=2005&sciodt=0,5&hl=en): 1,393)

<h3 id="seq2seq_based">Seq2Seq Based Methods</h3>

* Huang, Ting-Hao (Kenneth) and Ferraro, Francis and Mostafazadeh, Nasrin and Misra, Ishan and Agrawal, Aishwarya and Devlin, Jacob and Girshick, Ross and He, Xiaodong and Kohli, Pushmeet and Batra, Dhruv and Zitnick, C. Lawrence and Parikh, Devi and Vanderwende, Lucy and Galley, Michel and Mitchell, Margaret. 2016. [Visual Storytelling](https://www.aclweb.org/anthology/N16-1147v1). In *Proceedings of ACL 2016*. ([Citation](https://scholar.google.com/scholar?cites=9965124402549083081&as_sdt=2005&sciodt=0,5&hl=en): 76)
* Melissa Roemmele. 2016. [Writing Stories with Help from Recurrent Neural Networks](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPDFInterstitial/11966/12271). In *Proceedings of AAAI 2016*. ([Citation](https://scholar.google.com/scholar?cites=9018102316841841981&as_sdt=2005&sciodt=0,5&hl=en): 13)
* Jain, Parag and Agrawal, Priyanka and Mishra, Abhijit and Sukhwani, Mohak and Laha, Anirban and Sankaranarayanan, Karthik. 2017. [Story Generation from Sequence of Independent Short Descriptions](https://arxiv.org/pdf/1707.05501.pdf). *arXiv preprint arXiv:1707.05501*. ([Citation](https://scholar.google.com/scholar?cites=14358226381084770135&as_sdt=2005&sciodt=0,5&hl=en): 10)
* Liu, Tianyu and Wang, Kexiang and Sha, Lei and Chang, Baobao and Sui, Zhifang. 2017. [Table-to-text Generation by Structure-aware Seq2seq Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/16599/16019). In *Proceedings of AAAI 2018*. ([Citation](https://scholar.google.com/scholar?cites=17393004222760593181&as_sdt=2005&sciodt=0,5&hl=en): 14)
* Fan, Angela and Lewis, Mike and Dauphin, Yann. 2018. [Hierarchical Neural Story Generation](http://aclweb.org/anthology/P18-1082). In *Proceedings of ACL 2018*. ([Citation](https://scholar.google.com/scholar?cites=6764929710004991388&as_sdt=2005&sciodt=0,5&hl=en): 18)
* Song, Linfeng and Zhang, Yue and Wang, Zhiguo and Gildea, Daniel. 2018. [A Graph-to-Sequence Model for AMR-to-Text Generation](http://aclweb.org/anthology/P18-1150). In *Proceedings of ACL 2018*. ([Ciation](https://scholar.google.com/scholar?cites=8568627526215419713&as_sdt=2005&sciodt=0,5&hl=en): 10)
* Martin, Lara J and Ammanabrolu, Prithviraj and Wang, Xinyu and Hancock, William and Singh, Shruti and Harrison, Brent and Riedl, Mark O. 2018. [Event Representations for Automated Story Generation with Deep Neural Nets](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17046/15769). In *Proceedings of AAAI 2018*. ([Citation](https://scholar.google.com/scholar?cites=9614219931725711912&as_sdt=2005&sciodt=0,5&hl=en): 30)
* Clark, Elizabeth and Ji, Yangfeng and Smith, Noah A. 2018. [Neural Text Generation in Stories Using Entity Representation as Contex](https://www.aclweb.org/anthology/N18-1204). In *Proceedings of NAACL 2018*. ([Citation](https://scholar.google.com/scholar?cites=14104683731447446967&as_sdt=2005&sciodt=0,5&hl=en): 7)
* Wiseman, Sam and Shieber, Stuart and Rush, Alexander. 2018. [Learning Neural Templates for Text Generation](http://aclweb.org/anthology/D18-1356). In *Proceedings of EMNLP 2018*. ([Citation](https://scholar.google.com/scholar?cites=2282906185491164916&as_sdt=2005&sciodt=0,5&hl=en): 5)
* Chaturvedi, Snigdha and Peng, Haoruo and Roth, Dan. 2018. [Story Comprehension for Predicting What Happens Next](http://www.aclweb.org/anthology/D17-1168). In *Proceedings of EMNLP 2018*. ([Citation](https://scholar.google.com/scholar?cites=6900730782989730577&as_sdt=2005&sciodt=0,5&hl=en): 15)
* Zhang, Yue and Liu, Qi and Song, Linfeng. 2018. [Sentence-State LSTM for Text Representation](http://aclweb.org/anthology/P18-1030). In *Proceedings of ACL 2018*. ([Citation](https://scholar.google.com/scholar?cites=17679938345108531171&as_sdt=2005&sciodt=0,5&hl=en): 5)
* Kezar, Lee. 2018. [Mixed Feelings: Natural Text Generation with Variable, Coexistent Affective Categories](http://www.aclweb.org/anthology/P18-3020). In *Proceedings of ACL 2018, Student Research Workshop*.
* Welleck, Sean and Brantley, Kianté and Daumé III, Hal and Cho, Kyunghyun. 2019. [Non-Monotonic Sequential Text Generation](https://arxiv.org/pdf/1902.02192). *arXiv preprint arXiv:1902.02192*. ([Citation](https://scholar.google.com/scholar?cites=16018486661840997659&as_sdt=2005&sciodt=0,5&hl=en): 1)

<h3 id="vae_based">Variational Autoencoder Based Methods</h3>

* Li, Jiwei and Luong, Thang and Jurafsky, Dan. 2015. [A Hierarchical Neural Autoencoder for Paragraphs and Documents](http://aclweb.org/anthology/P15-1107). In *Proceedings of ACL 2015*. ([Citation](https://scholar.google.com/scholar?cites=8614969202104458050&as_sdt=2005&sciodt=0,5&hl=en): 283)
* Semeniuta, Stanislau and Severyn, Aliaksei and Barth, Erhardt. 2017. [A Hybrid Convolutional Variational Autoencoder for Text Generation](https://arxiv.org/pdf/1702.02390). In *Proceedings of EMNLP 2017*. ([Citation](https://scholar.google.com/scholar?cites=22686147435504388&as_sdt=2005&sciodt=0,5&hl=en): 57)
* Serban, Iulian Vlad and Ororbia II, Alexander and Pineau, Joelle and Courville, Aaron. 2017. [Piecewise Latent Variables for Neural Variational Text Processing](http://aclweb.org/anthology/W17-4308). In *Proceedings of EMNLP 2017*. ([Citation](https://scholar.google.com/scholar?cites=17428157778918133456&as_sdt=2005&sciodt=0,5&hl=en): 11)
* Yang, Zichao and Hu, Zhiting and Salakhutdinov, Ruslan and Berg-Kirkpatrick, Taylor. 2017. [Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](https://arxiv.org/pdf/1702.08139). In *Proceedings of ICML 2017*. ([Citation](https://scholar.google.com/scholar?cites=123823476347966018&as_sdt=2005&sciodt=0,5&hl=en): 72)
* Hu, Zhiting and Yang, Zichao and Liang, Xiaodan and Salakhutdinov, Ruslan and Xing, Eric P. 2017. [Toward Controlled Generation of Text](https://arxiv.org/pdf/1703.00955). In *Proceedings of ICML 2017*. ([Citation](https://scholar.google.com/scholar?cites=14533919283203963154&as_sdt=2005&sciodt=0,5&hl=en): 120)
* Deng, Yuntian and Kim, Yoon and Chiu, Justin and Guo, Demi and Rush, Alexander. 2018. [Latent Alignment and Variational Attention](https://papers.nips.cc/paper/8179-latent-alignment-and-variational-attention.pdf). In *Proceedings of NIPS 2018*. ([Citation](https://scholar.google.com/scholar?cites=6335407498429393003&as_sdt=2005&sciodt=0,5&hl=en&scioq=Incorporating+Structured+Commonsense+Knowledge+in+Story+Completion): 9)
* Kim, Yoon and Wiseman, Sam and Miller, Andrew C and Sontag, David and Rush, Alexander M. 2018. [Semi-Amortized Variational Autoencoders](https://arxiv.org/pdf/1802.02550.pdf). In *Proceedings of ICML 2018*. ([Citation](https://scholar.google.com/scholar?cites=15696369664604442539&as_sdt=2005&sciodt=0,5&hl=en): 27)
* Bahuleyan, Hareesh and Mou, Lili and Vechtomova, Olga and Poupart, Pascal. 2018. [Variational Attention for Sequence-to-Sequence Models](http://aclweb.org/anthology/C18-1142). In *Proceedings of COLING 2018*. ([Citation](https://scholar.google.com/scholar?cites=1653411252630135531&as_sdt=2005&sciodt=0,5&hl=en): 14)
* Xu, Jiacheng and Durrett, Greg. 2018. [Spherical Latent Spaces for Stable Variational Autoencoders](https://aclweb.org/anthology/D18-1480). In *Proceedings of EMNLP 2018*. ([Citation](https://scholar.google.com/scholar?cites=4068019460537216592&as_sdt=2005&sciodt=0,5&hl=en): 6)


<h3 id="gan_based">Generative Adversarial Nets Based Methods</h3>

* Kusner, Matt J and Hernández-Lobato, José Miguel. 2016. [GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution](https://arxiv.org/pdf/1611.04051). *arXiv preprint arXiv:1611.04051*. ([Citation](https://scholar.google.com/scholar?cites=11002066464238691420&as_sdt=2005&sciodt=0,5&hl=en): 71)
* Gulrajani, Ishaan and Ahmed, Faruk and Arjovsky, Martin and Dumoulin, Vincent and Courville, Aaron C. 2017. [Improved Training of Wasserstein GANs](https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf). In *Proceedings of NIPS 2017*. ([Citation](https://scholar.google.com/scholar?cites=3068694056154618633&as_sdt=2005&sciodt=0,5&hl=en): 1,102)
* Yu, Lantao and Zhang, Weinan and Wang, Jun and Yu, Yong. 2017. [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14344/14489). In *Proceedings of AAAI 2017*. ([Citation](https://scholar.google.com/scholar?cites=13783508915327278077&as_sdt=2005&sciodt=0,5&hl=en): 436)
* Liang, Xiaodan and Hu, Zhiting and Zhang, Hao and Gan, Chuang and Xing, Eric P. 2017. [Recurrent Topic-Transition GAN for Visual Paragraph Generation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liang_Recurrent_Topic-Transition_GAN_ICCV_2017_paper.pdf). In *Proceedings of IEEE 2017*. ([Citation](https://scholar.google.com/scholar?cites=7127182007926953895&as_sdt=2005&sciodt=0,5&hl=en): 65)
* Zhang, Yizhe and Gan, Zhe and Fan, Kai and Chen, Zhi and Henao, Ricardo and Shen, Dinghan and Carin, Lawrence. 2017. [Adversarial Feature Matching for Text Generation](https://arxiv.org/pdf/1706.03850). In *Proceedings of ICML 2017*. ([Citation](https://scholar.google.com/scholar?cites=11561684801033759674&as_sdt=2005&sciodt=0,5&hl=en): 68)
* Guo, Jiaxian and Lu, Sidi and Cai, Han and Zhang, Weinan and Yu, Yong and Wang, Jun. 2017. [Long Text Generation via Adversarial Training with Leaked Information](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/16360/16061). In *Proceedings of AAAI 2018*. ([Citation](https://scholar.google.com/scholar?cites=10032525507167574810&as_sdt=2005&sciodt=0,5&hl=en): 46)
* Xu, Jingjing and Ren, Xuancheng and Lin, Junyang and Sun, Xu. 2018. [Diversity-Promoting GAN: A Cross-Entropy Based Generative Adversarial Network for Diversified Text Generation](http://www.aclweb.org/anthology/D18-1428). In *Proceedings of EMNLP 2018*. ([Citation](https://scholar.google.com/scholar?cites=7729303371257026386&as_sdt=2005&sciodt=0,5&hl=en): 2)
* Mroueh, Youssef and Li, Chun-Liang and Sercu, Tom and Raj, Anant and Cheng, Yu. 2018. [Sobolev GAN](https://arxiv.org/abs/1711.04894). In *Proceedings of ICLR 2018*. ([Citation](https://scholar.google.com/scholar?cites=16587521411741023583&as_sdt=2005&sciodt=0,5&hl=en): 22)
* Fedus, William and Goodfellow, Ian and Dai, Andrew M. 2018. [MaskGAN: Better Text Generation via Filling in the_](https://arxiv.org/pdf/1801.07736). In *Proceedings of ICLR 2018*. ([Citation](https://scholar.google.com/scholar?cites=8054442901795858629&as_sdt=2005&sciodt=0,5&hl=en): 58) 

<h3 id="rl_based">Reinforcement Learning Based Methods</h3>

* Lin, Kevin and Li, Dianqi and He, Xiaodong and Zhang, Zhengyou and Sun, Ming-Ting. 2017. [Adversarial Ranking for Language Generation](http://papers.nips.cc/paper/6908-adversarial-ranking-for-language-generation.pdf). In *Proceedings of NIPS 2017*. ([Citation](https://scholar.google.com/scholar?cites=6871069604642164772&as_sdt=2005&sciodt=0,5&hl=en): 54)
* Che, Tong and Li, Yanran and Zhang, Ruixiang and Hjelm, R Devon and Li, Wenjie and Song, Yangqiu and Bengio, Yoshua. 2017. [Maximum-Likelihood Augmented Discrete Generative Adversarial Networks](https://arxiv.org/pdf/1702.07983). *arXiv preprint arXiv:1702.07983*. ([Citation](https://scholar.google.com/scholar?cites=15378466307857672293&as_sdt=2005&sciodt=0,5&hl=en): 64)
* Xu, Jingjing and Zhang, Yi and Zeng, Qi and Ren, Xuancheng and Cai, Xiaoyan and Sun, Xu. 2018. [A Skeleton-Based Model for Promoting Coherence Among Sentences in Narrative Story Generation](https://arxiv.org/pdf/1808.06945). In *Proceedings of EMNLP 2018*. ([Citation](https://scholar.google.com/scholar?cites=12919425531135632296&as_sdt=2005&sciodt=0,5&hl=en): 4)
* Wang, Xin and Chen, Wenhu and Wang, Yuan-Fang and Wang, William Yang. 2018. [No Metrics Are Prefect: Adversarial Reward Learning for Visual Storytelling](http://aclweb.org/anthology/P18-1083). In *Proceedings of ACL 2018*. ([Citation](https://scholar.google.com/scholar?cites=13450499082659265456&as_sdt=2005&sciodt=0,5&hl=en): 19)
* Hjelm, R Devon and Jacob, Athul Paul and Che, Tong and Trischler, Adam and Cho, Kyunghyun and Bengio, Yoshua. 2018. [Boundary-Seeking Generative Adversarial Networks](https://arxiv.org/pdf/1702.08431). In *Proceedings of ICLR 2018*. ([Citation](https://scholar.google.com/scholar?cites=7554343861717834099&as_sdt=2005&sciodt=0,5&hl=en&scioq=Incorporating+Structured+Commonsense+Knowledge+in+Story+Completion): 52)
* Shi, Zhan and Chen, Xinchi and Qiu, Xipeng and Huang, Xuanjing. 2018. [Towards Diverse Text Generation with Inverse Reinforcement Learning](https://arxiv.org/pdf/1804.11258). In *Proceedings of IJCAI 2018*. ([Citation](https://scholar.google.com/scholar?cites=10639156588114407661&as_sdt=2005&sciodt=0,5&hl=en): 4)

<h3 id="kbs_based">Knowledge Based Methods</h3>

* Liu, Hugo and Singh, Push. 2002. [MAKEBELIEVE: Using Commonsense Knowledge to Generate Stories](https://www.aaai.org/Papers/AAAI/2002/AAAI02-146.pdf). In *Proceedings of AAAI 2002*. ([Citation](https://scholar.google.com/scholar?cites=8122308501089511186&as_sdt=2005&sciodt=0,5&hl=en): 86)
* Yang, Bishan and Mitchell, Tom. 2017. [Leveraging Knowledge Bases in LSTMs for Improving Machine Reading](http://aclweb.org/anthology/P17-1132). In *Proceedings of ACL 2017*. ([Citation](https://scholar.google.com/scholar?cites=2883250958357294100&as_sdt=2005&sciodt=0,5&hl=en): 36)
* Ghazvininejad, Marjan and Brockett, Chris and Chang, Ming-Wei and Dolan, Bill and Gao, Jianfeng and Yih, Wen-tau and Galley, Michel. 2018. [A Knowledge-Grounded Neural Conversation Model](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/16710/16057). In *Proceedings of AAAI 2018*. ([Citation](https://scholar.google.com/scholar?cites=10303927314409501955&as_sdt=2005&sciodt=0,5&hl=en): 61)
* Li, Qian and Li, Ziwei and Wei, Jin-Mao and Gu, Yanhui and Jatowt, Adam and Yang, Zhenglu. 2018. [A Multi-Attention Based Neural Network with External Knowledge for Story Ending Predicting Task](http://www.aclweb.org/anthology/C18-1149). ([Citation](https://scholar.google.com/scholar?cites=223037972284583941&as_sdt=2005&sciodt=0,5&hl=en&scioq=Incorporating+Structured+Commonsense+Knowledge+in+Story+Completion): 2)
* Jian Guan, Yansen Wang and Minlie Huang. 2019. [Story Ending Generation with Incremental Encoding and Commonsense Knowledge](https://arxiv.org/pdf/1808.10113). In *Proceedings of AAAI 2019*. ([Citation](https://scholar.google.com/scholar?cites=13136733550716745125&as_sdt=2005&sciodt=0,5&hl=en): 3)
* Chen, Jiaao and Chen, Jianshu and Yu, Zhou. 2019. [Incorporating Structured Commonsense Knowledge in Story Completion](https://arxiv.org/pdf/1811.00625). In *Proceedings of AAAI 2019*.
* Shang, Mingyue and Fu, Zhenxin and Yin, Hongzhi and Tang, Bo and Zhao, Dongyan and Yan, Rui. 2019. [Find a Reasonable Ending for Stories: Does Logic Relation Help the Story Cloze Test?](https://arxiv.org/pdf/1812.05411). In *Student Abstract of AAAI 2019*.

# Reviews of KDD

**Official Review of Submission2677 by Reviewer YwPq**

**W1** Generalizability: The focus on Ethereum and TRON may limit the generalizability of the results to other blockchain platforms. Future work is needed to validate the framework's applicability across different blockchain ecosystems.

**Response** 
(1) Tron and Ethereum are the two most mainstream blockchain ecosystems in the current crypto gambling sector.
(2) Since both Tron and Ethereum use an EVM-compatible account model, and our graphical modeling is based on such an account model, our framework is not only applicable to Ethereum and Tron but also to other blockchain ecosystems based on the EVM account model, such as Arbitrum, Polygon, etc. The EVM is one of the most widespread and popular execution environments in current blockchain technology and applications.
(3) We have already expanded our data analysis to the Arbitrum chain. The relevant data acquisition, processing, and analysis results have all been made public on GitHub.
(4) In the case studies, one Ethereum platform had 274 nodes and 903 edges; one Tron Dapp transaction network had 340 nodes and 44,071 edges; one Arbitrum Dapp had 2,436 nodes and 65,545 edges, proving that our framework can also effectively handle the analysis of other similar networks.

**W2** Complexity and Scalability: The complexity of the CCDM framework and its scalability to larger blockchain ecosystsems is not thoroughly discussed. Addressing these aspects would strengthen the work.

**Response** 
The CCDM framework primarily includes two components: ChainVoteRank and Hidden Role Identification.
## ChainVoteRank
- **Edge Traversal and Node Weight Initialization**: 
  - Complexity: `O(|\tilde{E}|)`
  - Description: This step involves traversing all edges and initializing node weights, where `|\tilde{E}|` is the number of edges.
- **Score Calculation per Iteration**:
  - Complexity: `O(N)`
  - Description: Each node's input and output scores are calculated in each iteration round, where `N` is the number of nodes.
- **Max Score Node Identification**:
  - Complexity: `O(N)`
  - Description: Identifying the node with the highest score in the current iteration.
- **Neighbor Voting Power Update**:
  - Complexity: `O(\langle k \rangle)`
  - Description: Updating the voting power of a node's neighbors, where `\langle k \rangle` is the average degree of the network.
- **Overall Complexity**:
  - Complexity: `O(|\tilde{E}| + Np\langle k \rangle)`
  - Description: Where `p` is the number of output nodes.

## Hidden Role Identification
- **EGAT Computation**:
  - Complexity: `O(LE) ~ O(E)`
  - Description: EGAT calculations and loss computations are conducted with a complexity dependent on the number of edges `E` and the model layers `L`.
- **TLSTM Computation**:
  - Complexity: `O(NT) ~ O(N)`
  - Description: TLSTM calculations and loss computations for each node, where `T` is the sequence length and `N` is the number of nodes in the graph.
- **SSL Loss Computation**:
  - Complexity: `O(c*k*k)`
  - Description: Where `c` is the number of clusters and `k` is the number of nodes per cluster.

### Conclusion
The CCDM framework exhibits near-linear time complexity in handling sparse graphs, demonstrating good scalability, especially for large blockchain ecosystems characterized by sparse graphs.


**W3** Lack of Comparative Analysis: While the paper claims superiority over existing methods, a detailed comparative analysis with state-of-the-art techniques in role identification within blockchain ecosystems is missing. 

**Response** 
In the Web3 domain, role identification methods are underutilized. Our paper introduces the 2023 research by Wu et al.,[Reference 37] proposing a method called Evolved PageRank with Local Community Detection(EPLCD). Our CCDM framework is similar to this method(Evolved Pagerank and community detection). We provide a detailed comparative analysis in Section 4.2 and Figure 4, highlighting our approach's advantages over EPLCD and other methods.


**W4** Practical Implications: The practical implications of implementing CCDM, especially in terms of computational resources and real-time applicability, are not discussed.

**Response** 
In one Ethereum case, ChainVoteRank ran for 238ms and Hidden Role Identification for 745ms; in one TRON case, ChainVoteRank ran for 1013ms and Hidden Role Identification for 2984ms; in one Arbitrum case, ChainVoteRank ran for 37186ms and Hidden Role Identification for 44532ms. These times show that both components scale linearly with the size of the network in Ethereum and TRON cases. ChainVoteRank is more efficient than Hidden Role Identification. Despite the higher node and edge count in Arbitrum, the processing times remain practical for real-world applications.

**Q1** How does CCDM perform in terms of scalability and computational efficiency, especially when applied to larger blockchain networks beyond Ethereum and TRON?

**Response** 
For Q1, please refer to the answers of W1, W2, and W4.

**Q2** Can the authors elaborate on the comparative performance of CCDM against existing state-of-the-art techniques in role identification within blockchain ecosystems?

**Response** 
For Q2, please refer to the answers of W3.

**Q3** What are the potential challenges in generalizing CCDM to other blockchain platforms, and how might these challenges be addressed?

**Response** 

The CCDM framework is mainly used on Ethereum and Tron, blockchain platforms employing the Ethereum Virtual Machine (EVM) account model, commonly found in sectors like crypto casinos. The EVM is among the most prevalent execution environments in blockchain technology. However, adoption challenges arise with non-EVM platforms using different account models, such as the Unspent Transaction Output (UTXO) model. 
Before graph modeling, the UTXO model should be converted to an account model using heuristic address clustering [1,2,3], a well-established method in blockchain for identifying and aggregating addresses belonging to the same entity.
[1]Chang T H, Svetinovic D. Improving bitcoin ownership identification using transaction patterns analysis[J]. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2018, 50(1): 9-20.
[2]Zhao Z, Wang J, Shi K, et al. Improving Address Clustering in Bitcoin by Proposing Heuristics[J]. IEEE Transactions on Network and Service Management, 2022, 19(4): 3737-3749.
[3]Zhao Z, Wang J, Shi K, et al. Improving Address Clustering in Bitcoin by Proposing Heuristics[J]. IEEE Transactions on Network and Service Management, 2022, 19(4): 3737-3749.

**Q4** Could the authors discuss the practical implications of implementing CCDM, particularly regarding computational resources required and its real-time applicability?

**Response** For Q4, please refer to the answers of W1, W2, W3.


<!-- Scope: 3: The work is somewhat relevant to the Research track of KDD and is of narrow interest to a sub-community

Novelty: 4: Average 

Technical Quality: 3: Below Average

Presentation Quality: 2: Average (it needs some effort to understand, but it should be ok after some editing)

Reproducibility: 2: Average (some information is missing, but that could be easily fixed in the camera-ready version)

Reviewer Confidence: 3: The reviewer is confident but not certain that the evaluation is correct -->


---------------------------------------------------------------------------------------------------------



**Official Review of Submission2677 by Reviewer FyKh**


**W1** It would enhance clarity to illustrate the functioning of crypto-gambling ecosystems through formalization or by providing protocol examples. Readers without prior knowledge about GambleFi might feel confused and struggle to grasp the full story intended by this paper.

**Response**
In the crypto gambling ecosystem, there are two primary forms of user interaction with smart contracts: traditional fund transfers and direct connections to personal on-chain addresses. These are formally described and exemplified below.

Form 1: Traditional Fund Transfer
User Registration and Deposit: Users create or sign up for an account on a gambling platform and transfer cryptocurrency from their personal wallets to the smart contract address of the platform, serving as stakes or gaming credits.
Game Selection and Betting: Users select a game and place bets on the gambling platform, with bet details sent to the smart contract via a transaction.
Smart Contract Processing: The smart contract executes the game logic based on the bet information received, including random number generation and winner calculation. It updates the user’s balance on the platform or directly transfers the winnings in cryptocurrency.
Withdrawal: Users request the withdrawal of their winnings or balance from the platform. The smart contract processes the withdrawal and transfers the corresponding cryptocurrency to the user’s personal wallet address.

Form 2: Direct Connection to Personal On-Chain Addresses
Integrated Wallet Interaction: Users interact directly with smart contracts via an integrated wallet, eliminating the need for transfer or deposit.
Game Selection and Betting: Users select a game and initiate a betting transaction through their wallet.
Immediate Smart Contract Processing: The smart contract immediately processes the game logic and directly sends any winnings to the user’s on-chain address.

Crypto protocol examples:
https://etherscan.io/address/0x046eee2cc3188071c02bfc1745a6b17c656e3f3d#code
https://polygonscan.com/address/0xA45abc5A7F236B93809bB3228dD6e0b267b26fC4#code
https://arbiscan.io/address/0x51e99A0D09EeCa8d7EFEc3062AC024B6d0989959#code


**W2** Providing specific summary statistics would improve the paper, offering readers a basic understanding of the dataset's characteristics.

**Response**
Next, we'll display more statistics about data.
| Metrics               | ETH   | TRON    |
|-----------------------|-------|---------|
| Number of nodes       | 467615  | 1287360 |
| Number of edges       | 2486531 | 4354765 |
| Average degree        | 10.63 | 5.30    |
| Clustering coefficient| 0.24  | 0.17    |
| Assortativity         | -0.44 | 0.27    |
| Density               | 0.0121| 0.382   |
| Start_time            | Sep-21-2021|Sep-26-2021|
| End_time              | Mar-17-2023|Oct-14-2023|


**Q1** Could you provide formalized descriptions or protocol examples to elucidate the functioning of crypto-gambling ecosystems, particularly within the context of GambleFi?

**Response** For Q1, please refer to the answers of W1

**Q2** Would you consider including specific summary statistics in your paper to provide readers with a clearer understanding of the dataset's characteristics?


**Response** For Q2, please refer to the answers of W2


<!-- Scope: 4: The work is relevant to the Research track of KDD and is of broad interest to the community

Novelty: 5: Above Average 

Technical Quality: 4: Average

Presentation Quality: 2: Average (it needs some effort to understand, but it should be ok after some editing)

Reproducibility: 2: Average (some information is missing, but that could be easily fixed in the camera-ready version)

Reviewer Confidence: 3: The reviewer is confident but not certain that the evaluation is correct -->


---------------------------------------------------------------------------------------------------------



**Official Review of Submission2677 by Reviewer xc5N**


**W1** Inadequate presentation of experimental results, including the absence of detailed dataset descriptors and a clear enumeration of data points, labels, and the time period covered.

**Response** 
Next, we'll display more statistics about data.
| Metrics               | ETH   | TRON    |
|-----------------------|-------|---------|
| Number of nodes       | 467615  | 1287360 |
| Number of edges       | 2486531 | 4354765 |
| Average degree        | 10.63 | 5.30    |
| Clustering coefficient| 0.24  | 0.17    |
| Assortativity         | -0.44 | 0.27    |
| Density               | 0.0121| 0.382   |
| Start_time            | Sep-21-2021|Sep-26-2021|
| End_time              | Mar-17-2023|Oct-14-2023|


**W2** The absence of a parameter sensitivity analysis for self-supervised learning clusters and loss coefficients, as well as an ablation study to assess the impact of features, limits the comprehensiveness of the study.

**Response** 

Parameter Analysis: Attri-GAT output feature dimension (out_channels) is set to 8. LeakyReLU negative slope is 0.2. Number of attention heads (heads) is 4. Dropout rate is 0.5. Learning rate is 0.001. Training epochs are 100. Batch size is 32. Maximum number of clusters explored during clustering (max_clusters) is 20. For the LSTM model, the hidden layer size (hidden_size) is set to 64. The maximum sequence length (max_seq_length), used for padding or truncating each node's time series, is set to 5.
We have conducted an ablation study to assess the impact of features.
**Table. An ablation study on Ethereum case to assess the impact of features**
| Manual statistical features | Network structural features | Temporal evolution features | Accuracy | F1-score | Recall |
|-----------------------------|-----------------------------|-----------------------------|----------|----------|--------|
| Y                           | Y                           | N                           | 57.2     | 59.8     | 63.6   |
| N                           | Y                           | Y                           | 55.9     | 58.4     | 52.1   |
| Y                           | N                           | Y                           | 56.1     | 59.2     | 52.8   |


**W3** Overall Assessment: The methodology part is sound; however, the experimental results are not adequate. A first fatal shortcoming is a lack of dataset descriptors. You mention your data coming from "trusted data sources" but fail to provide a comprehensive list. I had never encountered the term "included but not limited to" in a research article before—let alone at a premier "data" science conference such as KDD. How many data points, how many labels, over what time period? These are all questions without an answer in the article.

**Response** 
Our reliable label data sources include dappradar.com, etherscan.io, tronscan.org, and arbiscan.io, as well as labels for Ethereum and TRON roles manually verified by judicial experts. Descriptive statistics for the data are provided in W1.

**W4** Earlier, 'a' was used to denote amount, but Algorithm 1 uses it to represent ability.

**Response** In Algorithm 1, we represent node voting abilities as 'ab_i' and 'ab_j' to distinguish them from the earlier 'a' (amount).


**W5** You need to define what a role is, how many there are, and whether a node could have multiple roles

**Response** 

We clearly define roles in our case study, and the number and proportion of roles are illustrated in Figure 3. In our research, a node’s position and function in the network structure are mapped to a specific role, allowing the model to focus on identifying the most prominent behavioral patterns and structural features of each node. Our goal is to discover and identify potential roles in the network without predefined role labels and categories. This approach, known as role discovery, fundamentally differs from soft clustering problems where roles or categories are predefined and known.

**W6** The figures suffer from low resolution, making them barely legible when printed.

**Response**
All images have been converted to PDF and uploaded to GitHub

**Q1** Self-supervised learning clusters are understudied. Do you consider each cluster as a role, or can a role have multiple clusters? How do you assign cluster labels (from limited, existing labels)?

**Response** 
(1) In our approach, each cluster obtained through self-supervised learning represents a potential role, assuming that nodes with similar features and behaviors share the same "role." Although complex roles may span multiple clusters, at this stage, each cluster is linked to a unique role.
(2) Our method primarily uses self-supervised learning to derive meaningful feature representations without relying on explicit cluster labels. The self-supervised learning loss function enables the model to autonomously explore and determine sample similarities, enhancing versatility and demonstrating its efficacy in label-scarce scenarios.

**Q2** Line 318: Should not the edge be represented as vi,vj,f,t, since this is a token transfer? If OG-IGM is multilayer, shouldn't edges have token identifiers (such as Tron, Storj, etc.)?

**Response** 
In line 318, we described edges in the form of vi, vj, f, t, where vi and vj represent the sender and receiver nodes of a transaction, f denotes the transaction amount, and t represents the time of the transaction. We model each gambling token separately, as different gambling token ecosystems may have unique participants, transaction patterns, and dynamics.

**Q3** What is deltaLtotal?

**Response** 

The term $|\Delta L_{Total}|$ refers to the change in the total loss function. This change is used to determine algorithm convergence during iterations. The iteration process terminates if $|\Delta L_{Total}| < p$, indicating convergence to a stable state.

**Q4** In line 564, how is similarity defined?（在第 564 行中，如何定义相似性？）自监督的损失函数的设计

**Response** 

The self-supervised loss function defines similarity based on distances between node embeddings. It minimizes intra-cluster distances and maximizes inter-cluster distances to learn an embedding space where similar nodes have closer embeddings and dissimilar nodes have farer embeddings.


<!-- Scope: 4: The work is relevant to the Research track of KDD and is of broad interest to the community

Novelty: 4: Average

Technical Quality: 3: Below Average

Presentation Quality: 2: Average (it needs some effort to understand, but it should be ok after some editing)

Reproducibility: 1: Poor (no code/data are given and important information is missing)

Reviewer Confidence: 3: The reviewer is confident but not certain that the evaluation is correct -->


---------------------------------------------------------------------------------------------------------



**Official Review of Submission2677 by Reviewer c5NA**

**W1** The preference for role-level recognition over address-level recognition warrants further clarification. Given that address-level recognition offers finer granularity, it would be insightful if the authors could elaborate on the advantages or unique insights gained from adopting a role-level approach.

**Response** 

Address-level identification could achieve finer-grained results, but only in labeled scenarios.
In reality, most blockchain transactions in anonymous address scenarios are unlabeled, facing many limitations when facing a new unknown scenario.
For unlabeled unknown scenarios lacking labels, role-level identification can first discover different behavioral entities (corresponding to roles in the ecosystem) through cluster analysis, distinguishing identities behind addresses in different clusters, and revealing common behavioral features behind the same cluster.
Role recognition results can also reveal information ignored by address-level identification (such as delegatee, influence of specific roles), and how these hidden roles negatively impact activities like gambling in encrypted casinos, including misleading, manipulation, fraud, market distortion and legal risks.
The results of role-level identification lay the foundation for understanding the blockchain gambling ecosystem, providing valuable insights for platform managers and regulators. For example, blockchain regulators can develop targeted regulatory policies based on identified roles, and gambling operators can design more reasonable game mechanisms based on player roles.


**W2** The constructed graph, as described in Section 3.1, focuses exclusively on numeric transaction records and event logs, overlooking the potential richness of semantic information inherent in the transactions. Incorporating or at least acknowledging this type of data could enhance the depth of analysis and the comprehensiveness of the graph model.

**Response** 

We acknowledge that in some blockchain application scenarios like smart contract interactions, rich semantic information does exist. However, in the GambleFi scenario we studied, transactions mainly represent the flow of funds, with relatively simple semantics mainly representing monetary transfers related to gambling activities. Therefore, we initially decided to focus our analysis on the numerical features of transactions to precisely capture this flow of funds. In future work, we plan to explore methods that combine numerical and semantic analysis to provide more comprehensive network analysis.

**W3** For readers to fully grasp the scope and significance of the experiments, it would be beneficial if the authors could provide detailed statistical information regarding the volume and characteristics of the data used, specifically for both Ethereum and TRON.（提供有关所用数据的数量和特征的详细统计信息，特别是针对以太坊和TRON的数据）

**Response** 

| Metrics               | ETH   | TRON    |
|-----------------------|-------|---------|
| Number of nodes       | 467615  | 1287360 |
| Number of edges       | 2486531 | 4354765 |
| Average degree        | 10.63 | 6.77    |
| Clustering coefficient| 0.24  | 0.17    |
| Assortativity         | -0.44 | 0.27    |
| Density               | 0.0121| 0.382   |
| Start_time            | Sep-21-2021|Sep-26-2021|
| End_time              | Mar-17-2023|Oct-14-2023|

Table: Network features of different roles on one Ethereum case. Network features include weighted in-degree($wd_{in}$), weighted out-degree($wd_{out}$), closeness centrality($cc$), betweenness centrality($bc$), authoritativeness($auth$), and hubness($hub$).
| Role               | $wd_{in}$ | $wd_{out}$ | $cc$  | $bc$    | $auth$ | $hub$  |
|--------------------|-----------|------------|-------|---------|--------|--------|
| Airdrop promotors  | 1.89      | 0.53       | 0.09  | 15.16   | 0.0001 | 0.0001 |
| Stakeholders       | 2.33      | 9.67       | 0.28  | 24.64   | 0.0002 | 0.0012 |
| Listing agents     | 2.05      | 1.92       | 0.30  | 196.16  | 0.0152 | 0.0013 |
| Arbitrageurs       | 2.77      | 2.61       | 0.29  | 450.40  | 0.0473 | 0.0019 |
| Real gamblers      | 1.87      | 1.05       | 0.15  | 72.08   | 0.0562 | 0.0006 |
| Exchanges          | 104.67    | 131.67     | 0.39  | 19016.8 | 0.0112 | 0.3811 |

Table: Account statistical features of different roles on one TRON Dapp, including the number of input transactions ($NIT$), the number of output transactions ($NOT$), the total amount of input transactions ($TAIT$) [Unit: USDT], and the total amount of output transactions ($TAOT$) [Unit: USDT].
| Addresses       | Roles               | $NIT$ | $NOT$  | $TAIT$       | $TAOT$       |
|-----------------|---------------------|-------|--------|--------------|--------------|
| TVXn6N...tXXXXX | Bonus payers        | 366   | 11,539 | 5,142,813.9  | 5,142,813.3  |
| TXTPLF...Y82zGH | Funding sponsors    | 3,654 | 3,755  | 10,020,873.5 | 10,040,463.5 |
| TYe2Kt...DRouGL | Disguised gamblers  | 6,947 | 9,494  | 14,299,163.9 | 14,298,197.1 |
| TUuvLo...eCGf4Z | Real gamblers       | 11    | 19     | 530.54       | 530.51       |
| TGDa2D...g9DDyU | Betting addresses   | 81    | 3      | 1,216.8      | 1,199.9      |
| TQpLsV...MMMMM  | Exchange addresses  | 87    | 4      | 1,994.9      | 1,991.9      |

**W4** The proposed model intriguingly integrates three distinct aspects of loss functions. To comprehensively evaluate the contribution of each component to the overall performance, conducting ablation studies would be highly valuable. Such studies would not only validate the necessity of each loss function aspect but also offer insights into their individual and combined impacts on the model's effectiveness.

**Response** 
**Table. An ablation study on Ethereum case to assess the impact of features**
| Manual statistical features | Network structural features | Temporal evolution features | Accuracy | F1-score | Recall |
|-----------------------------|-----------------------------|-----------------------------|----------|----------|--------|
| Y                           | Y                           | N                           | 57.2     | 59.8     | 63.6   |
| N                           | Y                           | Y                           | 55.9     | 58.4     | 52.1   |
| Y                           | N                           | Y                           | 56.1     | 59.2     | 52.8   |


**Q1** What are the preference for role-level recognition over address-level recognition? The constructed graph neglected potential richness of semantic information. Please provide detailed statistical information regarding the volume and characteristics of the data used Please conduct detailed ablation studies

**Response**


<!-- Scope: 2: The connection to KDD is weak

Novelty: 1: Very Poor 

Technical Quality: 2: Poor

Presentation Quality: 2: Average (it needs some effort to understand, but it should be ok after some editing)

Reproducibility: 2: Average (some information is missing, but that could be easily fixed in the camera-ready version)

Reviewer Confidence: 4: The reviewer is certain that the evaluation is correct and very familiar with the relevant literature -->


---------------------------------------------------------------------------------------------------------



**Official Review of Submission2677 by Reviewer 9sqm**


**W1** The authors endeavor to identify the Delegatees, who bring negative impacts on casinos and players by misleading players, manipulating the market, and committing fraud. However, the approach presented by the authors revolves around mining and characterizing patterns associated with various roles within casinos, ultimately falling short in effectively pinpointing the Delegatees.

**Response** 
Our study was indeed able to accurately identify several key roles within casinos, including Airdrop promoters, Stakeholders, Listing agents, Arbitrageurs, Real gamblers, Exchanges, Funding sponsors, Disguised gamblers, and Bonus payers. Figure 3 shows the identified role for each address. Due to length limitations, representative addresses were also outlined for each role on the TRON blockchain.

**W2** In contrast to mainstream DeFi applications, the gambling games hold a comparatively negligible market share. For instance, the Total Value Locked (TVL) across the entire DeFi ecosystem surpasses 95 billion USD (https://defillama.com/), whereas the cumulative TVL of all gambling games combined falls short of 100 thousand USD (https://alphagrowth.io/projects/top-gambling-projects-by-tvl). This disparity underscores the limited significance of the author's research. Therefore, it is better for the authors to shift their focus towards investigating the security of the broader DeFi ecosystem, rather than dedicating attention to individual obscure DeFi applications that contribute less than 0.0001% to the whole DeFi ecosystem's market share.

**Response**

While DeFi Llama's public data shows total locked value (TVL) for gambling games is relatively low, we noticed many gambling-related underground activities and applications may not be fully reflected in public data as DeFi Llama only tracks activities on the surface. Gambling is often part of the gray/black markets in crypto (https://www.unodc.org/roseap/en/2024/casinos-casinos-cryptocurrency-underground-banking/story.html). These underground activities are unlikely to report their true scale on public channels like DeFi Llama. Money laundering through gambling platforms could be considerably large but hidden underneath.
Cryptocurrency gambling behaviors involve complex transactions and interactions that may pose significant challenges in terms of security and compliance. With a lack of effective regulation currently for crypto gambling activities, research in this domain helps uncover potential risks and support policymaking.
From advertisements on blockchain explorers like Etherscan, we observed gambling platforms may actually have considerable user traffic and profits despite their seemingly low public TVL, indicating gambling could have a disproportionate impact on some aspects of the DeFi ecosystem.


**W3** Within existing literature, there is a noticeable absence of specific references to Delegatees in gambling games and their associated negative impacts. Despite vague acknowledgments of the presence of Delegatees and their potential adverse effects on gambling games, the authors' argument would benefit from substantiation through credible sources of information. Providing such sources would enhance the persuasiveness of the authors' claims and foster greater conviction among readers regarding the significance of the identified issue for gambling Delegatees.

**Response**

We further supplemented the following literature and research to support our views on the presence and impact of delegates in gambling games:
"Delegation and Public Pressure in a Threshold Public Goods Game" found that delegation, coupled with public pressure, significantly impacts contributions in public goods games, which can be analogous to strategies in gambling games.
“Detecting shill bidding in online English auctions” used observing bidding patterns in online auctions to detect the presence of shill bidding, an approach also applicable to identifying manipulative delegates in gambling games. 
"Observable Contracts: Strategic Delegation and Cooperation" discussed how strategic delegation can be used by players to commit to certain actions in a game, providing insights into how delegates can influence game outcomes, such as in gambling.


**W4** The authors' reliance on labeled data sourced directly from specific websites to construct their datasets raises concerns regarding the authenticity and reliability of the data. However, it's crucial to recognize that the labeled data obtained from these selected information sources inherently carries limitations and does not guarantee 100% reliability. In light of this, the authors should take proactive measures to enhance the reliability and effectiveness of their datasets. These measures are indispensable for upholding the credibility and integrity of the research outcomes.

**Response** 

For gambling DApp platforms, we not only referenced public information from specific platforms, but also directly accessed these platforms and observed their business operations to validate on-chain transaction data.
For Ethereum labels, we obtained transaction and role information from the authoritative blockchain data source Etherscan. For TRON labels, similar to the TRON data source, labels were obtained through cooperation with law enforcement agencies and provided official authentication of roles. This validation from official institutions ensured the authority and reliability of role labels, as mentioned in Section 4.1.

**W5** In the experimental case studies of Ethereum, despite the authors' efforts to cluster participation roles in gambling games into distinct categories such as Airdrop promoters, Listing agents, Arbitrageurs, and Real gamblers, the results of the experiments failed to identify any Delegatees that contribute negatively to gambling games. This outcome inadvertently underscores the limitations of the author's approach, suggesting that merely clustering gambling participants and categorizing role behaviors does not facilitate the discovery of Delegatees. Thus, further exploration of alternative approaches is necessary to effectively uncover and address the presence of Delegates in gambling contexts.

**Response** 
(1) Both Arbitrageurs and Disguised gamblers we identified could negatively impact gambling games. While arbitrageurs earn profits from price differences across markets, manipulating games or exploiting information asymmetries for arbitrage could be harmful. Disguised gamblers under fake identities may directly hurt games through deception, manipulation or fraud. Moreover, identifying Delegatees was not just about finding direct harm, but indirect influence on participants through various means.

(2) Our study was not limited to directly identifying casino participants. We analyzed on-chain entities related to the overall gambling platform ecosystem and gambling tokens, including gamblers, funders, promoters, etc. This crypto gambling transaction data aims to reveal a more holistic participant network and interactions within the gambling system.


**Q1** What efforts do the authors plan to take to address the mentioned cons?

**Response** Please refer to W1，W2，W3，W4，W5


<!-- Scope: 3: The work is somewhat relevant to the Research track of KDD and is of narrow interest to a sub-community

Novelty: 4: Average 

Technical Quality: 3: Below Average

Presentation Quality: 2: Average (it needs some effort to understand, but it should be ok after some editing)

Reproducibility: 1: Poor (no code/data are given and important information is missing)

Reviewer Confidence: 4: The reviewer is certain that the evaluation is correct and very familiar with the relevant literature -->


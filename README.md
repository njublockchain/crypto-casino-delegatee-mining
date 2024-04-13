# CCDM: Crypto Casino Delegatee Miner

Our proposed CCDM method not only identifies two fundamental roles, namely exchanges, and stakeholders but also uncovers hidden roles, such as airdrop promoters, listing agents, arbitrageurs, funding sponsors, and disguised gamblers.

## Ready

0. get the code ready `git clone https://github.com/njublockchain/gamblefi-role-identification && cd gamblefi-role-identification`
1. download and extract the dataset as [the instruction](./dataset/README.md)
2. ready the python env `conda create -n py39 python=3.9 && conda activate py39`
3. install the dependencies `pip install -r requirements.txt`


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


### Statistics about data.
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


**Table: Network features of different roles on one Ethereum case. Network features include weighted in-degree($wd_{in}$), weighted out-degree($wd_{out}$), closeness centrality($cc$), betweenness centrality($bc$), authoritativeness($auth$), and hubness($hub$).**
| Role               | $wd_{in}$ | $wd_{out}$ | $cc$  | $bc$    | $auth$ | $hub$  |
|--------------------|-----------|------------|-------|---------|--------|--------|
| Airdrop promotors  | 1.89      | 0.53       | 0.09  | 15.16   | 0.0001 | 0.0001 |
| Stakeholders       | 2.33      | 9.67       | 0.28  | 24.64   | 0.0002 | 0.0012 |
| Listing agents     | 2.05      | 1.92       | 0.30  | 196.16  | 0.0152 | 0.0013 |
| Arbitrageurs       | 2.77      | 2.61       | 0.29  | 450.40  | 0.0473 | 0.0019 |
| Real gamblers      | 1.87      | 1.05       | 0.15  | 72.08   | 0.0562 | 0.0006 |
| Exchanges          | 104.67    | 131.67     | 0.39  | 19016.8 | 0.0112 | 0.3811 |

**Table: Account statistical features of different roles on one TRON Dapp, including the number of input transactions ($NIT$), the number of output transactions ($NOT$), the total amount of input transactions ($TAIT$) [Unit: USDT], and the total amount of output transactions ($TAOT$) [Unit: USDT].**
| Addresses       | Roles               | $NIT$ | $NOT$  | $TAIT$       | $TAOT$       |
|-----------------|---------------------|-------|--------|--------------|--------------|
| TVXn6N...tXXXXX | Bonus payers        | 366   | 11,539 | 5,142,813.9  | 5,142,813.3  |
| TXTPLF...Y82zGH | Funding sponsors    | 3,654 | 3,755  | 10,020,873.5 | 10,040,463.5 |
| TYe2Kt...DRouGL | Disguised gamblers  | 6,947 | 9,494  | 14,299,163.9 | 14,298,197.1 |
| TUuvLo...eCGf4Z | Real gamblers       | 11    | 19     | 530.54       | 530.51       |
| TGDa2D...g9DDyU | Betting addresses   | 81    | 3      | 1,216.8      | 1,199.9      |
| TQpLsV...MMMMM  | Exchange addresses  | 87    | 4      | 1,994.9      | 1,991.9      |


**Table. An ablation study on Ethereum case to assess the impact of features**
| Manual statistical features | Network structural features | Temporal evolution features | Accuracy | F1-score | Recall |
|-----------------------------|-----------------------------|-----------------------------|----------|----------|--------|
| Y                           | Y                           | N                           | 57.2     | 59.8     | 63.6   |
| N                           | Y                           | Y                           | 55.9     | 58.4     | 52.1   |
| Y                           | N                           | Y                           | 56.1     | 59.2     | 52.8   |
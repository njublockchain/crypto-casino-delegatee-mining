# CCDM: Crypto Casino Delegatee Miner
Title: Gamblers or delegatees: Identifying hidden participant roles in the crypto gambling ecosystem
Abstract: With the development of blockchain technology, crypto gambling offers a greater level of secrecy due to its anonymity, garnering widespread popularity compared to traditional gambling. However, the operation of these crypto casinos is actually controlled by a small number of behind-the-scenes delegatees. These delegatees, hidden among players, cannot be easily distinguished and traced. In this paper, we firstly systematically focus on the identification of two key gamblers and delegatees roles within the crypto gambling ecosystem. Inspired by voting-style interaction patterns between participants, we propose a novel node voting method with a hierarchical clustering algorithm to map crypto addresses into potential roles. Experiments on real cases of Ethereum and TRON blockchain platforms demonstrate that our proposed approach not only identifies fundamental roles and gamblers but also uncovers diverse delegatees like airdrop promoters, listing agents, funding sponsors, arbitrageurs, and disguised gamblers. Remarkably, our results achieve a higher match with identities confirmed by judicial authorities than existing methods, which indicates the effectiveness of our approach in crypto gambling scenarios.

Our proposed CCDM method not only identifies two fundamental roles, namely exchanges, and stakeholders but also uncovers hidden roles, such as airdrop promoters, listing agents, arbitrageurs, funding sponsors, and disguised gamblers.

## Ready

0. get the code ready `git clone https://github.com/njublockchain/gamblefi-role-identification && cd gamblefi-role-identification`
1. download and extract the dataset as [the instruction](./dataset/README.md)
2. ready the python env `conda create -n py39 python=3.9 && conda activate py39`
3. install the dependencies `pip install -r requirements.txt`

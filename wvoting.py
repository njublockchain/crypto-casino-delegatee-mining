import math
import numpy as np

class ChainVoteRank:
    def __init__(self, nodeFile, n, discount, ability=None):
        self.N = n # Number of nodes
        self.nodeDegrees = [0] * self.N # Node degrees
        self.votes = [1.0] * self.N if ability is None else [ability] * self.N # Initial vote values
        self.voted_nums = [0.0] * self.N # Number of votes received 
        self.selected_nodes = [False] * self.N # Flag for selected nodes
        self.nodeAdj = [[] for _ in range(self.N)] # Adjacency list
        self.weights = np.zeros((self.N, self.N)) # Edge weights
        self.discount = discount # Vote discount factor

        # Read graph from file
        with open(nodeFile, 'r') as f:
            lines = f.readlines()

        for line in lines:
            terms = line.split()
            term_1 = int(terms[0]) - 1
            term_2 = int(terms[1]) - 1
            term_3 = float(terms[2])

            self.nodeAdj[term_1].append(term_2) # Add edge
            self.nodeAdj[term_2].append(term_1)
            self.nodeDegrees[term_1] += 1 # Update degrees
            self.nodeDegrees[term_2] += 1
            self.weights[term_1][term_2] = term_3 # Add edge weight
            self.weights[term_2][term_1] = term_3

    def findMax(self):
        # Find unselected node with max votes
        max_i = 0
        max_val = 0.0

        for i in range(self.N):
            if self.voted_nums[i] > max_val and not self.selected_nodes[i]:
                max_val = self.voted_nums[i]
                max_i = i

        return max_i

    # Modified version to exclude spreader neighbors
    def findMax_plus(self, spreader_neighbours):
        max_i = 0
        max_val = 0.0

        for i in range(self.N):
            if self.voted_nums[i] >= max_val and not self.selected_nodes[i]:
                if (i + 1) not in spreader_neighbours:
                    max_val = self.voted_nums[i]
                    max_i = i

        return max_i

    def vote_process_plus(self, topk, debug=False):
        
        # Voting process with spreader neighbor exclusion
        
        votedResults = [0] * self.N
        spreader_neighbors = {} 

        j = 0

        while topk > 0:
            
            voting_number = 0
            
            # Calculate votes
            for i in range(self.N):
                if not self.selected_nodes[i]:
                    neighbours = self.nodeAdj[i]
                    sum_val = 0.0
                    for nei in neighbours:
                        if not self.selected_nodes[nei]:
                            sum_val = self.votes[nei] * self.weights[i][nei] + sum_val
                            voting_number += 1

                    self.voted_nums[i] = math.pow(sum_val * len(neighbours), 0.5)

            if voting_number == 0:
                break

            # Select node with max votes
            max_i = self.findMax_plus(spreader_neighbors) 

            # Debug prints
            if debug:
                print("----------------------------")
                print("voting ability:")
                print(" ".join(map(str, self.votes)))
                print("voted numbers:")
                print(" ".join(map(str, self.voted_nums)))
                print("----------------------------")

            if max_i >= 0:
                self.selected_nodes[max_i] = True
                self.votes[max_i] = 0
                sel_neighbours = self.nodeAdj[max_i]
                for nei in sel_neighbours:
                    if self.votes[nei] - self.discount >= 0:
                        self.votes[nei] = self.votes[nei] - self.discount
                    else:
                        self.votes[nei] = 0.0
                    spreader_neighbors[nei] = nei
                votedResults[j] = max_i + 1
                j += 1
                topk -= 1

            self.voted_nums = [0] * self.N
        
        # Fill remaining votedResult slots with 0
        for k in range(j, self.N):
            votedResults[k] = 0

        return votedResults

    def vote_process(self, topk):
      
        # Original voting process
        
        votedResults = [0] * self.N

        j = 0

        while topk > 0:
          
            voting_number = 0
          
            # Calculate votes
            for i in range(self.N):
                if not self.selected_nodes[i]:
                    neighbours = self.nodeAdj[i]
                    sum_val = 0.0
                    for nei in neighbours:
                        if not self.selected_nodes[nei]:
                            sum_val = self.votes[nei] * self.weights[i][nei] + sum_val
                            voting_number += 1

                    self.voted_nums[i] = math.pow(sum_val * len(neighbours), 0.5)

            if voting_number == 0:
                break

            # Select node with max votes 
            max_i = self.findMax_plus()
            self.selected_nodes[max_i] = True
            self.votes[max_i] = 0
            sel_neighbours = self.nodeAdj[max_i]
            for nei in sel_neighbours:
                if self.votes[nei] - self.discount >= 0:
                    self.votes[nei] = self.votes[nei] - self.discount
                else:
                    self.votes[nei] = 0.0

            votedResults[j] = max_i + 1
            j += 1
            topk -= 1

            self.voted_nums = [0] * self.N
        
        # Fill remaining votedResult slots with 0
        for k in range(j, self.N):
            votedResults[k] = 0

        return votedResults

if __name__ == "__main__":
    fileInput = "./dataset/wvoting_data/0xc2a.txt" 
    vote_rank = ChainVoteRank(fileInput, 273, 0.303)
    topk = vote_rank.vote_process_plus(20, True)
    print(topk)

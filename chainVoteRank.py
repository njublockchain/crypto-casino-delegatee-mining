import csv

class ChainVoteRank:
    def __init__(self, nodeFile, n, discount):
        self.N = n # Number of nodes
        self.votes_in = [1.0] * self.N # In-link vote values
        self.votes_out = [1.0] * self.N # Out-link vote values  
        self.voted_nums_in = [0.0] * self.N # In-link votes received
        self.voted_nums_out = [0.0] * self.N # Out-link votes received
        self.selected_nodes = [False] * self.N # Flag for selected nodes
        self.nodeAdj = [[] for _ in range(self.N)] # Adjacency list
        self.weights = [[0.0] * self.N for _ in range(self.N)] # Edge weights
        self.discount = discount # Vote discount factor

        self.initializeData()
        self.readFile(nodeFile)

    def initializeData(self):
        # Initialize all data structures
        for i in range(self.N):
            self.votes_in[i] = 1.0  
            self.votes_out[i] = 1.0
            self.voted_nums_in[i] = 0.0
            self.voted_nums_out[i] = 0.0
            self.selected_nodes[i] = False
            self.nodeAdj[i] = []

            for j in range(self.N):
                self.weights[i][j] = 0.0

    def readFile(self, nodeFile):
        # Read graph from file
        with open(nodeFile, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                term_1 = int(row[0])
                term_2 = int(row[1])
                term_3 = float(row[2])

                self.nodeAdj[term_1].append(term_2) 
                self.weights[term_1][term_2] = term_3

    def findMax(self, voted_nums):
        # Find unselected node with max votes
        max_i = 0
        max_val = 0.0

        for i in range(self.N):
            if voted_nums[i] > max_val and not self.selected_nodes[i]:
                max_val = voted_nums[i]
                max_i = i

        return max_i

    def vote_process(self, topk):
        
        votedResults = [0] * self.N

        j = 0
        while topk > 0:
            
            # Calculate in and out votes
            for i in range(self.N):
                if not self.selected_nodes[i]:
                    neighbours = self.nodeAdj[i]
                    sum_in = 0.0
                    sum_out = 0.0

                    for nei in neighbours:
                        if not self.selected_nodes[nei]:
                            sum_out += self.votes_out[nei] * self.weights[i][nei]

                    for k in range(self.N):
                        if not self.selected_nodes[k] and self.weights[k][i] != 0:
                            sum_in += self.votes_in[k] * self.weights[k][i]

                    self.voted_nums_in[i] = pow(sum_in * len(neighbours), 0.5)
                    self.voted_nums_out[i] = pow(sum_out * len(neighbours), 0.5)

            # Get max in and out nodes
            max_i_in = self.findMax(self.voted_nums_in)  
            max_i_out = self.findMax(self.voted_nums_out)

            # Select node and update votes
            if self.voted_nums_in[max_i_in] > self.voted_nums_out[max_i_out]:
                self.updateVotes(max_i_in, self.votes_in)
                votedResults[j] = max_i_in
            else:
                self.updateVotes(max_i_out, self.votes_out)
                votedResults[j] = max_i_out

            topk -= 1
            j += 1

        return votedResults

    def updateVotes(self, idx, votes):
        # Update votes after node selection
        self.selected_nodes[idx] = True

        neighbours = self.nodeAdj[idx]
        for nei in neighbours:
            if not self.selected_nodes[nei]:
                if idx in self.nodeAdj[nei]:
                    self.votes_in[nei] *= self.discount 
                else:
                    self.votes_out[nei] *= self.discount
                    
if __name__ == '__main__':
    # Run me
    nodeFile = ""  
    n = 274  
    discount = 0.6  
    topK = 274

    wvr = ChainVoteRank(nodeFile, n, discount)
    wvr.vote_process(topK)

    print("Node\tIn-Vote\tOut-Vote")
    for i in range(n):
        print(f"{i}\t{wvr.voted_nums_in[i]}\t{wvr.voted_nums_out[i]}")
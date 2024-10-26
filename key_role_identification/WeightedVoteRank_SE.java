package influential_spreaders;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class WeightedVoteRank_SE {
    int N;
    double discount;
    double[] votes_in;
    double[] votes_out;
    double[] voted_nums_in;
    double[] voted_nums_out;
    boolean[] selected_nodes;
    double[][] weights;
    List<List<Integer>> nodeAdj;
    double[] inDegreeEntropy;  // 存储每个节点的入度结构熵
    double[] outDegreeEntropy; // 存储每个节点的出度结构熵

    public WeightedVoteRank_SE(String nodeFile, int n, double discount) throws IOException {
        this.N = n;
        this.votes_in = new double[this.N];
        this.votes_out = new double[this.N];
        this.voted_nums_in = new double[this.N];
        this.voted_nums_out = new double[this.N];
        this.selected_nodes = new boolean[this.N];
        this.nodeAdj = new ArrayList<>(this.N);
        this.weights = new double[this.N][this.N];
        this.discount = discount;

        this.inDegreeEntropy = new double[this.N];
        this.outDegreeEntropy = new double[this.N];

        initializeData();
        readFile(nodeFile);
        calculateStructureEntropy();  // 计算每个节点的入度和出度结构熵
        writeEntropyToFile("/Users/mac/IdeaProjects/Voting/node_entropy.txt");  // 将结构熵写入文件
        initializeVotes();  // 使用结构熵初始化投票能力
    }

    private void initializeData() {
        for (int i = 0; i < this.N; i++) {
            this.voted_nums_in[i] = 0.0;
            this.voted_nums_out[i] = 0.0;
            this.selected_nodes[i] = false;
            this.nodeAdj.add(new ArrayList<>());

            for (int j = 0; j < this.N; j++) {
                this.weights[i][j] = 0.0;
            }
        }
    }

    private void readFile(String nodeFile) throws IOException {
        FileReader fr = new FileReader(nodeFile);
        BufferedReader br = new BufferedReader(fr);

        String line = br.readLine();

        while (line != null) {
            String[] terms = line.split(" ");
            int term_1 = Integer.parseInt(terms[0]);
            int term_2 = Integer.parseInt(terms[1]);
            double term_3 = Double.parseDouble(terms[2]);

            this.nodeAdj.get(term_1).add(term_2);
            this.weights[term_1][term_2] = term_3;

            line = br.readLine();
        }
        br.close();
    }

    // 计算每个节点的入度和出度结构熵
    // 计算每个节点的入度和出度结构熵
//    private void calculateStructureEntropy() {
//        int totalEdges = 0;  // 边的总数
//
//        // 计算图中边的总数
//        for (int i = 0; i < this.N; i++) {
//            for (int j = 0; j < this.N; j++) {
//                if (this.weights[j][i] > 0) {
//                    totalEdges += 1;  // 只计数边的数量
//                }
//            }
//        }
//
//        System.out.println("Total edges: " + totalEdges); // 打印总边数
//
//        // 计算每个节点的入度和出度结构熵
//        for (int i = 0; i < this.N; i++) {
//            int inDegree = 0;  // 节点i的入度
//            int outDegree = 0; // 节点i的出度
//
//            for (int j = 0; j < this.N; j++) {
//                if (this.weights[j][i] > 0) {
//                    inDegree += 1;  // 统计入度
//                }
//                if (this.weights[i][j] > 0) {
//                    outDegree += 1;  // 统计出度
//                }
//            }
//
//            if (inDegree > 0) {
//                inDegreeEntropy[i] = -(double)inDegree / totalEdges * Math.log((double)inDegree / totalEdges) / Math.log(2);
//            } else {
//                inDegreeEntropy[i] = 0.0;
//            }
//
//            if (outDegree > 0) {
//                outDegreeEntropy[i] = -(double)outDegree / totalEdges * Math.log((double)outDegree / totalEdges) / Math.log(2);
//            } else {
//                outDegreeEntropy[i] = 0.0;
//            }
//
//        }
//    }

    private void calculateStructureEntropy() {
        double totalWeight = 0.0;  // V_G: 图中所有边的权重之和

        // 计算总权重和每个节点的加权入度和出度
        double[] weightedInDegree = new double[this.N];
        double[] weightedOutDegree = new double[this.N];
        for (int i = 0; i < this.N; i++) {
            for (int j = 0; j < this.N; j++) {
                double weight = this.weights[i][j];
                if (weight > 0) {
                    totalWeight += weight;
                    weightedOutDegree[i] += weight;
                    weightedInDegree[j] += weight;
                }
            }
        }

        System.out.println("Total edge weight: " + totalWeight);

        // 计算每个节点的入度和出度结构熵
        for (int i = 0; i < this.N; i++) {
            if (weightedInDegree[i] > 0) {
                double p = weightedInDegree[i] / totalWeight;
                inDegreeEntropy[i] = -p * (Math.log(p) / Math.log(2));
            } else {
                inDegreeEntropy[i] = 0.0;
            }

            if (weightedOutDegree[i] > 0) {
                double p = weightedOutDegree[i] / totalWeight;
                outDegreeEntropy[i] = -p * (Math.log(p) / Math.log(2));
            } else {
                outDegreeEntropy[i] = 0.0;
            }
        }
    }


    private void writeEntropyToFile(String filename) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
            writer.write("Node\tInDegreeEntropy\tOutDegreeEntropy\n");
            for (int i = 0; i < this.N; i++) {
                writer.write(i + "\t" + inDegreeEntropy[i] + "\t" + outDegreeEntropy[i] + "\n");
            }
            writer.close();
            System.out.println("File " + filename + " generated successfully."); // 添加成功提示
        } catch (IOException e) {
            e.printStackTrace();  // 打印异常信息
        }
    }

    // 初始化投票能力，根据结构熵初始化
    private void initializeVotes() {
        for (int v = 0; v < this.N; v++) {
            this.votes_in[v] = this.inDegreeEntropy[v];
            this.votes_out[v] = this.outDegreeEntropy[v];
        }
    }

    public int findMax(double[] voted_nums) {
        int max_i = 0;
        double max = 0.0;

        for (int i = 0; i < this.N; i++) {
            if (voted_nums[i] > max && this.selected_nodes[i] == false) {
                max = voted_nums[i];
                max_i = i;
            }
        }

        return max_i;
    }

    public int[] vote_process(int topk) {
        long startTime = System.currentTimeMillis();
        int[] votedResults = new int[this.N];

        int j = 0;
        while (topk > 0) {
            for (int i = 0; i < N; i++) {
                if (selected_nodes[i] == false) {
                    double sum_in = 0.0;
                    double sum_out = 0.0;
                    int in_neighbours_count = 0;
                    int out_neighbours_count = 0;

                    // 计算 in-score 和入边邻居数量
                    for (int k = 0; k < N; k++) {
                        if (this.selected_nodes[k] == false && this.weights[k][i] != 0) {
                            sum_in += this.votes_out[k] * this.weights[k][i];
                            in_neighbours_count++;
                        }
                    }

                    // 计算 out-score 和出边邻居数量
                    List<Integer> out_neighbours = nodeAdj.get(i);
                    for (Integer nei : out_neighbours) {
                        if (this.selected_nodes[nei] == false) {
                            sum_out += this.votes_in[nei] * this.weights[i][nei];
                            out_neighbours_count++;
                        }
                    }

                    // 使用不同的邻居数量计算 in-score 和 out-score
                    this.voted_nums_in[i] = Math.pow(sum_in * in_neighbours_count, 0.5);
                    this.voted_nums_out[i] = Math.pow(sum_out * out_neighbours_count, 0.5);
                }
            }

            int max_i_in = findMax(voted_nums_in);
            int max_i_out = findMax(voted_nums_out);

            if (voted_nums_in[max_i_in] > voted_nums_out[max_i_out]) {
                updateVotes(max_i_in, votes_in);
                votedResults[j] = max_i_in;
            } else {
                updateVotes(max_i_out, votes_out);
                votedResults[j] = max_i_out;
            }

            topk--;
            j++;
        }

        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Total runtime of the voting process: " + totalTime + " ms");

        return votedResults;
    }


    private void updateVotes(int idx, double[] votes) {
        this.selected_nodes[idx] = true;

        List<Integer> neighbours = nodeAdj.get(idx);
        for (Integer nei : neighbours) {
            if (this.selected_nodes[nei] == false) {
                if (nodeAdj.get(nei).contains(idx)) {
                    votes_in[nei] = votes_in[nei] * this.discount;
                } else {
                    votes_out[nei] = votes_out[nei] * this.discount;
                }
            }
        }
    }

    public static void main(String[] args) {
        String nodeFile = "/Users/mac/IdeaProjects/Voting/Tron_hanhua_graph_1.txt";  // 更换为您的网络数据文件路径
        int n = 340;  // 更换为您网络中的节点数量
        double discount = 0.1;  // 设置合适的折扣因子
        int topK = 39;  // 设置要获取的前K个节点

        try {
            WeightedVoteRank_SE wvr = new WeightedVoteRank_SE(nodeFile, n, discount);
            wvr.vote_process(topK);

            System.out.println("Node\tIn-Vote\tOut-Vote");
            for (int i = 0; i < n; i++) {
                System.out.println(i + "\t" + wvr.voted_nums_in[i] + "\t" + wvr.voted_nums_out[i]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
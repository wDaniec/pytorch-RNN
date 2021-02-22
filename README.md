# recurent-relational-network
pyTorch implementation of: https://arxiv.org/abs/1711.08028 with additional features. The network can be trained on kakuro puzzles.
Link to the thesis: [link](https://www.researchgate.net/publication/344417396_Solving_Kakuro_Problems_using_Recurrent_Relational_Networks)

To create datasets I used an existing repository: https://github.com/Aditya239/Kakurosy

The modifiction is avaiable: https://github.com/wDaniec/Kakurosy

Every change I made can be seen in the last commit. Function solve() in file testuje.py was written by Aditya239. The function was modified so that LPSolver can solve instances which are of size greater than 9x9.

"""Module for learning graphs from smooth signals"""
# from https://github.com/LTS4/graph-learning

from itertools import combinations_with_replacement
from typing import Optional, Tuple, Union, Dict

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator


def sum_squareform(n: int) -> Tuple[sparse.csr_array, sparse.csr_array]:
    """For *z* vectorform of *Z*, return the operator S such that S @ z =
    np.sum(Z, axis=-1) and its transpose"""
    # pylint: disable=invalid-name
    nnz = n * (n - 1)

    J = np.concatenate([i * np.ones(n - 1) for i in range(n)])

    slices = []
    offsets = []
    start = 0
    stop = 0
    for i in range(n):
        offsets.append([sl[i - j - 1] for j, sl in enumerate(slices)])
        stop = start + n - i - 1
        slices.append(list(range(start, stop)))
        start = stop

    I = np.concatenate([off + sl for sl, off in zip(slices, offsets)])

    # # offset
    # k = 0
    # for i in range(2, n):
    #     I[k : k + (n - i)] = np.arange(i, n)
    #     k = k + (n - i + 1)

    # St = sparse([1:ncols, 1:ncols], [I, J], 1, ncols, n)
    St = sparse.coo_array(
        (np.ones(nnz), (I, J)),
        shape=(nnz // 2, n),
    )
    return St.T.tocsr(), St.tocsr()


def gsp_learn_graph_log_degrees(
    distances: np.ndarray,
    alpha: float,
    beta: float,
    edge_init: Optional[np.ndarray] = None,
    maxit: int = 1000,
    tol: float = 1e-5,
    step_size: float = 0.5,
    edge_tol: float = 1e-3,
) -> np.ndarray:
    r"""Computes a weighted adjacency matrix $W$ from squared pairwise distances
    in $Z$, using the smoothness assumption that $\text{trace}(X^TLX)$ is small,
    where $X$ is the data (columns) changing smoothly from node to node on the
    graph and $L = D-W$ is the combinatorial graph Laplacian. See the paper of
    the references for the theory behind the algorithm.


    Args:
        distances (np.ndarray): Matrix with (squared) pairwise distances of
            nodes, or its vectorization.
        alpha (float): Log prior constant  (bigger a -> bigger weights in W)
        beta (float): :math:`\|W\|_F^2` prior constant  (bigger b -> more dense W)
        edge_init (Optional[np.ndarray]): Initialization point. default:
            `zeros_like(distances)`
        maxit (int): Maximum number of iterations. Default: 1000
        tol (float): Tolerance for stopping criterion. Defaul: 1e-5
        step_size (float): Step size from the interval (0,1). Default: 0.5

    Returns:
        np.ndarray: Weighted adjacency matrix


    Alternatively, Z can contain other types of distances and use the
    smoothness assumption that

    .. sum(sum(W .* Z))

    .. math:: \sum_i\sum_j W_{ij}Z_{ij}

    is small.

    The minimization problem solved is

    .. minimize_W sum(sum(W .* Z)) - a * sum(log(sum(W))) + b * ||W||_F^2/2 + c
    * ||W-W_0||_F^2/2

    .. math:: \min_W \sum_i\sum_j W_{ij}Z_{ij} -\alpha\sum_i\log(\sum_jW_{ij}) +
        \frac{\beta}{2} \|W\|_F^2 + c/2 ||W - W0||_F^2

    subject to $W$ being a valid weighted adjacency matrix (non-negative,
    symmetric, with zero diagonal).

    The algorithm used is forward-backward-forward (FBF) based primal dual
    optimization (see references).

    The stopping criterion is whether both relative primal and dual
    distance between two iterations are below a given tolerance.

    To set the step size use the following rule of thumb: Set it so that
    relative change of primal and dual converge with similar rates.

    Adapted from matlab implementation in `GSP toolbox`_ by William Cappelletti

    References:
    - V. Kalofolias and N. Perraudin, “Large Scale Graph Learning from Smooth
        Signals.” arXiv, May 01, 2019. doi: 10.48550/arXiv.1710.05654.
    - V. Kalofolias, “How to Learn a Graph from Smooth Signals,” in Proceedings
        of the 19th International Conference on Artificial Intelligence and
        Statistics, May 2016, pp. 920–929. doi: 10.48550/arXiv.1601.02513.
    - N. Komodakis and J.-C. Pesquet, “Playing with Duality: An overview of
        recent primal?dual approaches for solving large-scale optimization
        problems,” IEEE Signal Processing Magazine, vol. 32, no. 6, pp. 31–54, Nov.
        2015, doi: 10.1109/MSP.2014.2377273.

    .. _`GSP toolbox`: https://epfl-lts2.github.io/gspbox-html/
    """
    # pylint: disable=too-many-locals

    n_edges = distances.shape[-1]
    n_dim = len(distances.shape)

    if n_dim == 2:
        distances = distances.T
    elif n_dim > 2:
        raise ValueError("Distances must be vector or matrix")

    sum_op, sum_op_t = sum_squareform(int(np.round((1 + np.sqrt(1 + 8 * n_edges)) / 2)))

    step_size /= 2 * beta + np.sqrt(
        2 * (sum_op.shape[0] - 1)  # This approximate sparse.linalg.norm(sum_op, ord=2)
    )

    # Variable and dual
    if edge_init is not None:
        edge_w = edge_init
    else:
        edge_w = np.zeros_like(distances)
    d_n = sum_op @ edge_w

    for i in range(maxit):
        y_n = (1 - step_size * 2 * beta) * edge_w - step_size * (sum_op_t @ d_n)
        yb_n = d_n + step_size * (sum_op @ edge_w)

        p_n = np.where((prox := y_n - 2 * step_size * distances) > 0, prox, 0)
        pb_n = (yb_n - np.sqrt(yb_n**2 + 4 * alpha * step_size)) / 2

        q_n = (1 - step_size * 2 * beta) * p_n - step_size * (sum_op_t @ pb_n)
        qb_n = pb_n + step_size * (sum_op @ p_n)

        if i > 0:
            # Denominators are previous iteration ones
            rel_norm_primal = np.linalg.norm(-y_n + q_n, ord=2) / np.linalg.norm(edge_w, ord=2)
            rel_norm_dual = np.linalg.norm(-yb_n + qb_n, ord=2) / np.linalg.norm(d_n, ord=2)
        else:
            rel_norm_primal = rel_norm_dual = np.inf

        edge_w = edge_w - y_n + q_n
        d_n = d_n - yb_n + qb_n

        if rel_norm_primal < tol and rel_norm_dual < tol:
            break

    edge_w[edge_w < edge_tol] = 0

    if n_dim == 1:
        return squareform(edge_w)
    else:
        return edge_w.T


def get_theta(
        sq_pdists: NDArray[np.float64],
        avg_degree: Union[int, Dict[Tuple[int, int], int]],
        blocks: NDArray[np.int64] = None,
) -> Union[float, NDArray[np.float64]]:

    # 函数实现
    """Compute a theta parameter to obtain desired average degree for the graph learning LogModel.

    The parametrization is from V. Kalofolias and N. Perraudin, “Large Scale
    Graph Learning from Smooth Signals.” arXiv, May 01, 2019. doi: 10.48550/arXiv.1710.05654.

    Args:
        sq_pdists (NDArray[np.float64]): Squared pairwise distances of the data.
        avg_degree (int | dict[tuple[int, int], int]): Desired average degree. If a
            dictionary is passed, it should map block pairs to average degrees.
            NOTE: Due to the standard parameterization not allowing for self-edges, off diagonal
            blocks might have a lower avg_degree than asked.
        blocks (NDArray[np.int64], optional): Vector of block assignments for each node.
            Defaults to None.

    Returns:
        float | NDArray[np.float64]: Theta parameter to rescale pairwise distances.
            If blocks are provided, returns the vectorized upper triangular matrix of
            theta parameters for each edge.
    """
    if blocks is None:
        sorted_dists = np.sort(sq_pdists, 1)
        partial_sum = sorted_dists[:, :avg_degree].sum(1)

        theta_min = np.mean(
            (
                avg_degree * sorted_dists[:, avg_degree] ** 2
                - partial_sum * sorted_dists[:, avg_degree]
            )
            ** (-0.5)
        )
        theta_max = np.mean(
            (
                avg_degree * sorted_dists[:, avg_degree - 1] ** 2
                - partial_sum * sorted_dists[:, avg_degree - 1]
            )
            ** (-0.5)
        )
        return np.sqrt(theta_min * theta_max).item()
    else:
        thetas = np.zeros(2 * (len(blocks),))

        for pair in combinations_with_replacement(np.unique(blocks).tolist(), 2):
            slice1 = np.where(blocks == pair[0])[0][:, np.newaxis]
            slice2 = np.where(blocks == pair[1])[0][np.newaxis, :]

            if isinstance(avg_degree, dict):
                block_degree = avg_degree[pair]
            else:
                block_degree = avg_degree

            theta = get_theta(sq_pdists[slice1, slice2], avg_degree=block_degree)
            thetas[slice1, slice2] = thetas[slice2.T, slice1.T] = theta

        return squareform(thetas, checks=False)


class LogModel(BaseEstimator):
    """Extract graph on which data is smooth from Kalofolias 2016 model.

    Uses the optimal parameters from Kalofolias 2019.

    Parameters:
        avg_degree (int, optional): Desired average number of neighbours for node. Defaults to None.
        edge_init (Optional[NDArray[np.float64]], optional): Prior on graph
            weights. Defaults to None.
        maxit (int, optional): Max optimizaiton iterations. Defaults to 1000.
        tol (float, optional): Optimization tolerance. Defaults to 1e-5.
        step_size (float, optional): Optimization step size. Defaults to 0.5.
        edge_tol (float, optional): Tolerance to keep positive edges. Defaults to 1e-3.

    Attributes:
        theta_ (float): Sparsity parameter to rescale pairwise distances.
        weights_ (NDArray[np.float64]): Edge weights of the learned graph.
    """

    def __init__(
        self,
        avg_degree: int = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        edge_init: Optional[NDArray[np.float64]] = None,
        maxit: int = 1000,
        tol: float = 1e-5,
        step_size: float = 0.5,
        edge_tol: float = 1e-3,
    ) -> None:
        self.avg_degree = avg_degree
        self.alpha = alpha
        self.beta = beta
        self.edge_init = edge_init
        self.maxit = maxit
        self.tol = tol
        self.step_size = step_size
        self.edge_tol = edge_tol

        self.theta_: float
        self.weights_: NDArray[np.float64]

    def _initialize(self, x) -> NDArray[np.float64]:
        sq_pdists = pdist(x.T) ** 2
        if self.avg_degree is None:
            self.theta_ = 1
        else:
            self.theta_ = get_theta(squareform(sq_pdists), self.avg_degree)

        return sq_pdists

    def fit(self, x: NDArray[np.float64]):
        sq_pdists = self._initialize(x)

        self.weights_ = gsp_learn_graph_log_degrees(
            sq_pdists * self.theta_,
            alpha = self.alpha,
            beta = self.beta,
            edge_init=self.edge_init,
            maxit=self.maxit,
            tol=self.tol,
            step_size=self.step_size,
            edge_tol=self.edge_tol,
        )

        return self


if __name__ == "__main__":

    # 直接使用内置函数的用法：
    # from scipy.spatial.distance import pdist, squareform
    # from graph_log_learn import gsp_learn_graph_log_degrees
    #
    # dist_vec = pdist(X_data.T) ** 2  # X_data.shape=(dim, n_nodes)
    # W = gsp_learn_graph_log_degrees(dist_vec, alpha=1.0, beta=1.0, ...)
    # 即可返回一个 [n_nodes, n_nodes] 的邻接矩阵。

    # 使用类的用法：
    # 1) 准备数据
    #   例如: X_data 形状 = [dim, n_nodes]
    #   这里我们假设有 50 维特征, 100 个节点
    dim, n_nodes = 50, 100
    rng = np.random.default_rng(seed=42)

    # 生成一些模拟数据
    # 比如说每个节点在 50 维空间的坐标
    X_data = rng.normal(loc=0, scale=1, size=(dim, n_nodes))

    # 2) 实例化模型
    #    设置想要的平均度为 5
    log_model = LogModel(
        avg_degree=10,
        alpha=1e-2,  # 默认值
        beta=1e-1,  # 默认值
        edge_init=None,  # 不用任何先验
        maxit=1000,  # 最大迭代
        tol=1e-5,  # 收敛阈值
        step_size=0.5,
        edge_tol=1e-3  # 边权门限,小于此值的边设置为0
    )

    # 3) 拟合模型
    log_model.fit(X_data)

    # 4) 获取学到的邻接矩阵
    W_learned = log_model.weights_
    print("Learned adjacency matrix shape:", W_learned.shape)

    # 5) 你可以查看平均度数是否接近5
    #    由于W_learned对角线=0, 且是对称, 可以用 np.count_nonzero
    #    或简单地计算每行非0元素的平均
    degrees = (W_learned > 0).sum(axis=1)
    avg_deg_est = degrees.mean()
    print("Estimated average degree:", avg_deg_est)

    # 6) 计算平滑项: trace(X^T L X)
    D = np.diag(W_learned.sum(axis=1))
    L = D - W_learned  # shape: [n_nodes, n_nodes]
    d_sqrt_inv = 1.0 / np.sqrt(np.diag(D) + 1e-9)  # shape=[n_nodes]
    D_sqrt_inv = np.diag(d_sqrt_inv)  # shape=[n_nodes,n_nodes]
    L_norm = D_sqrt_inv @ L @ D_sqrt_inv  # shape=[n_nodes,n_nodes]

    #    注意 X_data.shape = [dim, n_nodes] => 转置为 [n_nodes, dim]
    X_nodes_dim = X_data.T  # [n_nodes, dim]
    smooth_val = np.trace(X_nodes_dim.T @ L_norm @ X_nodes_dim)
    scale_val = np.trace(X_nodes_dim.T @ X_nodes_dim)  # 信号能量

    smooth_normalized = smooth_val / scale_val
    print(f"Smoothness value for learned adjacency: {smooth_normalized:.4f}")
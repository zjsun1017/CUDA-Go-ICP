# Fast Globally Optimal ICP - A CUDA Implementation of [Go-ICP](https://github.com/yangjiaolong/Go-ICP)

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Final Project**

* **Authors**: Mufeng Xu & Zhaojin Sun


## Algorithm
### Iterative Closest Point (ICP)
The Iterative Closest Point (ICP) algorithm is a widely used method for aligning two point clouds by iteratively minimizing the distance between corresponding points. It computes the optimal rigid transformation (rotation and translation) to align the source point cloud to the target. A more detailed pipeline is shown below.

#### 1. Initialization
- Define two point clouds:
  - Source point cloud: $P = \{p_1, p_2, \dots, p_n\} \in \mathbb{R}^3$
  - Target point cloud: $Q = \{q_1, q_2, \dots, q_m\} \in \mathbb{R}^3$
- Initialize the transformation matrix $T = [R|t]$, where $R$ is the rotation matrix and $t$ is the translation vector.

#### 2. Find Closest Points
For each point $p_i \in P$, find the closest point $q_j \in Q$ using the Euclidean distance:
$q_j = \arg\min_{q \in Q} \|p_i - q\|_2$. Create a set of corresponding pairs $(p_i, q_j)$.

#### 3. Compute Transformation Using Procrustes Method

To compute the optimal transformation $T = (R, t)$ that minimizes the alignment error:
$E(R, t) = \sum_{i=1}^{n} \|Rp_i + t - q_i\|_2^2$.

- **Step 1: Center the Points**
Compute centroids:
$\bar{p} = \frac{1}{n} \sum_{i=1}^{n} p_i, \quad \bar{q} = \frac{1}{n} \sum_{i=1}^{n} q_i$.
Center the point clouds:
$P_c = \{p_i - \bar{p}\}, \quad Q_c = \{q_i - \bar{q}\}$.

- **Step 2: Compute the Cross-Covariance Matrix**
Compute the covariance matrix $H$:
$= \sum_{i=1}^{n} P_c[i] Q_c[i]^\top$.

- **Step 3: Perform Singular Value Decomposition (SVD)**
Decompose $H$ using SVD:
$H = U \Sigma V^\top$.
Compute the rotation matrix $R$:
$R = V U^\top$.
Ensure $R$ is a proper rotation matrix with $\det(R) = 1$.

- **Step 4: Compute the Translation Vector**
Compute $t$:
$= \bar{q} - R \bar{p}$.

#### 4. Apply Transformation
Apply the transformation to the source point cloud:
$P \leftarrow \{Rp_i + t , p_i \in P\}$.

#### 5. Evaluate Convergence
Compute the mean squared alignment error:
$E = \frac{1}{n} \sum_{i=1}^{n} \|Rp_i + t - q_i\|_2^2$. Check convergence:
- If $E$ is below a threshold $\epsilon$, or the change in $E$ between iterations is small, terminate.
- Otherwise, repeat steps 2–5.

#### Final Output
Return the final transformation $T = (R, t)$ and the aligned point cloud $P$.


<!-- Idea to describe the algorithm:
- Draw a contour of a rough terrain with lots of local minima
- Draw grids 
- Draw arrows indicating the steps of ICP, which marches towards the local minima
- Eliminate grids that have lower bound worse than the optimal error now
- Draw finer grids and repeat
 -->

### Globally Optimal ICP (Go-ICP)
The key algorithm in the globally optimal ICP problem is **Branch-and-Bound** (BnB),
global minimum is searched in the $SE(3)$ space (3D Rigid Rotation and Translation), which is 6-dimensional.

Instead of directly searching over $SE(3)$, it is proposed to launch an outer-BnB,
which searches over the $SO(3)$ space (3D Rotation) with BnB algorithm,
and for each branch, an inner-BnB function, which searches over the $\mathbb{R}^3$ space,
will be invoked. Below are the detailed steps:

#### 1. Initialization
- Define two point clouds:
  - Source point cloud: $P = \{p_1, p_2, \dots, p_n\} \in \mathbb{R}^3$
  - Target point cloud: $Q = \{q_1, q_2, \dots, q_m\} \in \mathbb{R}^3$
- Initialize the transformation space $\mathcal{T}$ to cover all possible transformations in SE(3):
  - Rotation space: $SO(3)$, represented as a unit quaternion or Euler angles.
  - Translation space: $\mathbb{R}^3$, bounded by the data range of $P$ and $Q$.
- Set an initial upper bound for the alignment error $E_{ub} = \infty$.

#### 2. Compute the Distance Metric
For each point in $P$, define the point-to-point distance metric as $d(p_i, Q) = \min_{q_j \in Q} \|p_i - q_j\|_2$. Precompute a distance field (e.g., KD-tree or voxel grid) for efficient nearest neighbor search in $Q$.

#### 3. Branch-and-Bound Framework

- **Step 1: Partition the Transformation Space**
Divide the transformation space $\mathcal{T}$ into smaller regions (branches), represented by parameter bounds for rotation and translation:
  - Rotation bounds: Define regions in $SO(3)$ using quaternion or angle intervals.
  - Translation bounds: Define regions in $\mathbb{R}^3$ as bounding boxes.

- **Step 2: Evaluate Lower Bound for Each Region**
For each region of the transformation space:
  - Compute a lower bound of the alignment error $E_{lb}$ by assuming the best possible alignment within the region.
  - Use precomputed distances from the distance field for efficient evaluation: $E_{lb} = \sum_{i=1}^{n} \min_{q_j \in Q} \|p_i - q_j\|_2^2$.

- **Step 3: Prune the Regions**
  - If $E_{lb} > E_{ub}$, discard the region (it cannot improve the current best solution).
  - Otherwise, keep the region for further exploration.

- **Step 4: Refine the Upper Bound**
  - If a region is sufficiently small, compute the exact transformation that minimizes the alignment error within this region using the Procrustes method.
  - Update the upper bound $E_{ub}$ if a better transformation is found: $E_{ub} = \min(E_{ub}, E_{region})$.

#### 4. Convergence Check
- Terminate the algorithm if the difference between $E_{ub}$ and $E_{lb}$ across all regions is smaller than a predefined threshold $\epsilon$.
- Otherwise, continue partitioning the remaining regions.

#### Final Output
- Return the globally optimal transformation $T = (R, t)$ and the aligned point cloud $P$.
- The alignment error $E_{opt} = E_{ub}$ is guaranteed to be the global minimum.

- Performance:
  - ICP is faster for well-aligned point clouds due to its simplicity and focus on local optimization.
  - Go-ICP is slower because it exhaustively searches both rotation and translation spaces but ensures the best possible result.
- Robustness:
  - Both ICP and Go-ICP are sensitive to noise and outliers, which can lead to poor results in challenging scenarios.

## Reference

1. Jiaolong Yang, Hongdong Li, Dylan Campbell and Yunde Jia. [Go-ICP: A Globally Optimal Solution to 3D ICP Point-Set Registration](https://arxiv.org/pdf/1605.03344). IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2016.   
2. Jiaolong Yang, Hongdong Li and Yude Jia. [Go-ICP: Solving 3D Registration Efficiently and Globally Optimally](https://openaccess.thecvf.com/content_iccv_2013/papers/Yang_Go-ICP_Solving_3D_2013_ICCV_paper.pdf). International Conference on Computer Vision (ICCV), 2013. 
3. [Go-ICP (GitHub)](https://github.com/yangjiaolong/Go-ICP)

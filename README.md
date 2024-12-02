# Fast Globally Optimal ICP - A CUDA Implementation of [Go-ICP](https://github.com/yangjiaolong/Go-ICP)

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Final Project**

* **Authors**: Mufeng Xu & Zhaojin Sun
* Tested on:
  * Windows 11, i9-13900H @ 2.6GHz 32GB, RTX 4080 Laptop 12GB (Personal Computer)
  * Windows 11, i9- @ ...GHz 64GB, RTX 4090 Laptop 16GB (Personal Computer)

## Introduction

This project implements CUDA acceleration for ICP: the classic point cloud registration algorithm, and its globally optimal improvement, Go-ICP. The project also provides tools for visualization and performance measurement.

## Demo
Here is a demo that compares the speed of original Go-ICP paper and our method.
<div style="display: flex; justify-content: space-around;">
  <img src="img/sgoicp.gif" alt="First GIF" width="400" />
  <img src="img/fgoicp.gif" alt="Second GIF" width="400" />
</div>

### How to build and run the code
**Step 1: Hardware Setup**
- Any modern CPU and an operating system of any modern distribution should be compatible with this project.
- You must have a GPU that supports CUDA version 12.6 or higher. CUDA version 11.8 is not sufficient. Here is the [list of CUDA-compatible GPUs](https://developer.nvidia.com/cuda-gpus) along with their compute capabilities. For instructions on installing the CUDA Toolkit, you can refer to the following website: [Cuda installation Guide](https://docs.nvidia.com/cuda/index.html).

**Step 2: Software Setup**
- You will need a version of CMake that supports the C++17 standard to configure the project.
- For Windows users, it is recommended to use Visual Studio 2019 or newer version and `x64` for your platform, as we do not include Win32 libraries. You can configure the project using either CMake's GUI or console commands.
- If you are on a Linux system, execute the following commands in the root directory:
```
mkdir build
cd build
cmake ..
make -j8
```
- The project uses OpenGL to provide visualization. In the `/external` folder located in the root directory, we have already included Windows binaries and headers for GLEW and GLFW. For Linux systems, ensure that the right libraries (dev versions) are installed for mesa and glx for your distribution.

**Step 3: Run the demo**
- The configuration `.toml` files are located in the `/test` folder in the root directory. Pass the file path as an argument to specify the configuration target. For example:
```./bin/cis5650_fgo_icp ../test/bunny.toml``` If you are using Visual Studio, make sure the corresponding path is added to the **Command Arguments** in the project configuration.
- Point cloud data (source: [The Stanford 3D Scanning Repository
](https://graphics.stanford.edu/data/3Dscanrep/), [Artec 3D](https://www.artec3d.com/3d-models/ply) and original paper) is stored in the `/data` folder in the root directory. Currently, the project supports two file formats: `.txt` and `.ply`. For the internal format, please refer to the point cloud files we have provided. Here is an [online viewer for `.ply` files](https://imagetostl.com/view-ply-online). 
- We have also provided a Python script: `transform_point_cloud.py` for you to generate data point clouds from existing ones.

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

### Comparison between ICP and Go-ICP
- Convergence
  - ICP converges to a local minimum, highly dependent on initial alignment and prone to suboptimal results if the initial guess is poor.
  - Go-ICP guarantees global convergence through a branch-and-bound framework, ensuring the optimal solution regardless of initial alignment.
  - **An example of ICP converging to local minima**

<div style="display: flex; justify-content: space-around;">
  <img src="img/localminima.gif" alt="First GIF" width="400" />
</div>


- Performance:
  - ICP is faster for well-aligned point clouds due to its simplicity and focus on local optimization.
  - Go-ICP is slower because it exhaustively searches both rotation and translation spaces but ensures the best possible result.
- Robustness:
  - Both ICP and Go-ICP are sensitive to noise and outliers, which can lead to poor results in challenging scenarios.


## Acceleration

### Accelerating ICP
**Sample point clouds in advance**
- During the point cloud loading phase, you can choose to sample the point cloud at a specific ratio: `subsample` in the `.toml` configuration file. This significantly improves processing speed. Since the current sampling method is entirely uniform, it generally does not lead to the ICP converging to another position compared to ICP without sampling.
- **A demo before and after sampling with ratio 0.05**
<div style="display: flex; justify-content: space-around;">
  <img src="img/nosample.png" alt="First" width="400" />
  <img src="img/sample0.05.png" alt="Second" width="400" />
</div>

**Parallelization of Procrustes Method**
- For each point in the point cloud, we utilize a CUDA kernel for parallelization. Specifically, this includes: applying mean centering to the point cloud, computing the covariance (i.e., outer product), and applying the $SE(3)$ homogeneous transformation to the point cloud.
- To accelerate all summation operations, we use thrust::reduce. This specifically includes calculating the mean of the point cloud and obtaining the covariance matrix from the covariance computation.
- **A demo of our ICP vs our CUDA-accelerated ICP**

<div style="display: flex; justify-content: space-around;">
  <img src="img/ICP.gif" alt="First GIF" width="400" />
  <img src="img/ICP_GPU.gif" alt="Second GIF" width="400" />
</div>

**Find Closest Points with k-d Tree**
- A k-d tree is a binary search tree for organizing points in k-dimensional space, optimized for tasks like nearest neighbor search and range queries. To use it, first build the tree by recursively splitting points along one dimension at a time (e.g., x, y, z) based on median values. For nearest neighbor queries, traverse the tree by comparing the query point with node split dimensions, prune regions that can't contain closer points, and backtrack as needed. This is a very brief introduction. For more information, you can read the [Wikipedia page on k-d trees](https://en.wikipedia.org/wiki/K-d_tree).

![k-d Tree Visualization](https://upload.wikimedia.org/wikipedia/commons/b/b6/3dtree.png)
- We integrated the k-d tree from the nanoflann library to accelerate nearest neighbor searches in the point cloud. However, for smaller point clouds, our tests revealed that the k-d tree is not faster than naive iteration.

### Accelerating Go-ICP
**Terminate Early**
- In the original paper, the termination condition for the search is that the lower bounds of all remaining search spaces exceed the current best error. However, in practice, we observed that the point cloud often converges to the correct alignment much earlier, but a significant amount of time is still wasted exploring potentially better parameters. To address this, we introduced an error threshold to terminate the search early.
- **Performance gain by terminating early**

<div style="display: flex; justify-content: space-around;">
  <img src="img/goicp.gif" alt="First GIF" width="400" />
  <img src="img/sgoicp.gif" alt="Second GIF" width="400" />
</div>

**Flattened k-d Tree on GPU**
- To improve the speed of the k-d tree, we attempted to flatten the k-d tree structure and store it in the GPU's memory. However, we found that this approach not only failed to enhance performance but actually made subsequent attempts to parallelize Go-ICP even slower (as shown in the speed comparison below)! We suspect this is due to the memory discontinuity introduced by flattening, along with some peculiar issues related to Thrust.
![kdperform.png](img%2Fkdperform.png)
- In conclusion, we ultimately confirmed that k-d trees are not suitable for CUDA and GPU acceleration. Except for the CPU mode of Go-ICP and one mode of ICP on GPU, all remaining k-d tree implementations have been removed and replaced with faster alternatives such as Lookup Tables or straightforward brute-force search.

**Lookup Table (LUT) on GPU**

A Lookup Table (LUT) is a data structure that precomputes and stores the results of computationally expensive operations, enabling rapid retrieval during runtime. In our application, we construct a 3D LUT to store squared distances to the nearest point in the target point cloud.

* Efficient Storage: The LUT is stored as a CUDA Texture Object, which allows for efficient lookups due to optimized memory access patterns.
* Parallel Construction: The LUT is built in parallel on the GPU, offering significantly faster performance compared to CPU-based construction, even with higher resolution.

This approach leverages the computational power of GPUs to enhance the speed and efficiency of nearest neighbor searches in 3D point clouds.

**Parallelization of Translation search**

**Search Heuristics**





## Dependencies

* [OpenGL Mathematics (GLM)](https://glm.g-truc.net/0.9.4/api/index.html)
* [Eigen 3](https://eigen.tuxfamily.org/index.php?title=Main_Page)
* [TOML++](https://github.com/marzer/tomlplusplus)
* [tinyply](https://github.com/ddiakopoulos/tinyply)
* [nanoflann](https://github.com/jlblancoc/nanoflann)
  
We've included `TOML++`, `tinyply`, and `nanoflann` in our repo. You can install `glm` and `Eigen` via `vcpkg` on Windows.

## Reference

1. Jiaolong Yang, Hongdong Li, Dylan Campbell and Yunde Jia. [Go-ICP: A Globally Optimal Solution to 3D ICP Point-Set Registration](https://arxiv.org/pdf/1605.03344). IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2016.   
2. Jiaolong Yang, Hongdong Li and Yude Jia. [Go-ICP: Solving 3D Registration Efficiently and Globally Optimally](https://openaccess.thecvf.com/content_iccv_2013/papers/Yang_Go-ICP_Solving_3D_2013_ICCV_paper.pdf). International Conference on Computer Vision (ICCV), 2013. 
3. [Go-ICP (GitHub)](https://github.com/yangjiaolong/Go-ICP)
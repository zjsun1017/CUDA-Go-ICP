# Fast Globally Optimal ICP - A CUDA Implementation of [Go-ICP](https://github.com/yangjiaolong/Go-ICP)

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Final Project**

* **Authors**: Mufeng Xu & Zhaojin Sun
* Tested on: 
    * Windows 11/Ubuntu 24.04, i9-13900H @ 2.6GHz 32GB, RTX 4080 Laptop 12GB (Personal Computer)


## Introduction

This project implements CUDA acceleration for ICP: the classic point cloud registration algorithm, and its globally optimal improvement, Go-ICP. The project also provides tools for visualization and performance measurement.


## Demo








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
- Point cloud data is stored in the `/data` folder in the root directory. Currently, the project supports two file formats: `.txt` and `.ply`. For the internal format, please refer to the point cloud files we have provided.


## Algorithm

<!-- Idea to describe the algorithm:
- Draw a contour of a rough terrain with lots of local minima
- Draw grids 
- Draw arrows indicating the steps of ICP, which marches towards the local minima
- Eliminate grids that have lower bound worse than the optimal error now
- Draw finer grids and repeat
 -->

### Iterative Closest Point (ICP)
The Iterative Closest Point (ICP) algorithm is a widely used method for aligning two point clouds by iteratively minimizing the distance between corresponding points. It computes the optimal rigid transformation (rotation and translation) to align the source point cloud to the target. A more detailed pipeline is shown below.


#### 1. Initialization
- Define two point clouds:
  - Source point cloud: $P = \{p_1, p_2, \dots, p_n\} \in \mathbb{R}^3$
  - Target point cloud: $Q = \{q_1, q_2, \dots, q_m\} \in \mathbb{R}^3$
- Initialize the transformation matrix $T = [R|t]$, where $R$ is the rotation matrix and $t$ is the translation vector.

#### 2. Find Closest Points
For each point $p_i \in P$, find the closest point $q_j \in Q$ using the Euclidean distance:
$ q_j = \arg\min_{q \in Q} \|p_i - q\|_2 $. Create a set of corresponding pairs $(p_i, q_j)$.

#### 3. Compute Transformation Using Procrustes Method

To compute the optimal transformation $T = (R, t)$ that minimizes the alignment error:
$ E(R, t) = \sum_{i=1}^{n} \|Rp_i + t - q_i\|_2^2$.

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
$t = \bar{q} - R \bar{p}$

#### 4. Apply Transformation
Apply the transformation to the source point cloud:
$P \leftarrow \{Rp_i + t \, | \, p_i \in P\}$

#### 5. Evaluate Convergence
Compute the mean squared alignment error:
$E = \frac{1}{n} \sum_{i=1}^{n} \|Rp_i + t - q_i\|_2^2$

Check convergence:
- If $E$ is below a threshold $\epsilon$, or the change in $E$ between iterations is small, terminate.
- Otherwise, repeat steps 2–5.

#### Final Output
Return the final transformation $T = (R, t)$ and the aligned point cloud $P$.







### Globally Optimal ICP (Go-ICP)
The key algorithm in the globally optimal ICP problem is **Branch-and-Bound** (BnB),
global minimum is searched in the $SE(3)$ space (3D Rigid Rotation and Translation), which is 6-dimensional.

Instead of directly searching over $SE(3)$, it is proposed to launch an outer-BnB,
which searches over the $SO(3)$ space (3D Rotation) with BnB algorithm,
and for each branch, an inner-BnB function, which searches over the $\mathbb{R}^3$ space,
will be invoked. 

## Our Works
### Accelerating ICP
### Accelerating Go-ICP



## Dependencies

- OpenGL Mathematics (GLM) 
- [TOML++](https://github.com/marzer/tomlplusplus)
- [tinyply](https://github.com/ddiakopoulos/tinyply)
- [nanoflann](https://github.com/jlblancoc/nanoflann)

## Reference

1. Jiaolong Yang, Hongdong Li, Dylan Campbell and Yunde Jia. [Go-ICP: A Globally Optimal Solution to 3D ICP Point-Set Registration](https://arxiv.org/pdf/1605.03344). IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2016.   
2. Jiaolong Yang, Hongdong Li and Yude Jia. [Go-ICP: Solving 3D Registration Efficiently and Globally Optimally](https://openaccess.thecvf.com/content_iccv_2013/papers/Yang_Go-ICP_Solving_3D_2013_ICCV_paper.pdf). International Conference on Computer Vision (ICCV), 2013. 
3. [Go-ICP (GitHub)](https://github.com/yangjiaolong/Go-ICP)
# AirSim
 AirSim是用于无人机，汽车等的场景模拟器，基于虚幻引擎（虚幻引擎4）。它是开源的，跨平台的，用于物理和视觉逼真的模拟。本文的目标是描述AirSim平台的搭建使用过程，以实验自动驾驶汽车的深度学习和计算机视觉。
 
1.课题介绍
在1956年, 美国的JohnMcCarthy提出AI之后, 越来越多的应用引入了人工智能这一学科。它是研究用于模拟人的智能的理论、方法、技术及应用系统的一门新的技术科学, 是将人的思想运用于其它方面的重要开发领域。自动驾驶：自动驾驶又被称为无人车、无人驾驶等 (autopilot, automatic driving, self-driving, driveless) 。对于自动驾驶的概念解释, 业界有着明确的等级划分, 可被它们分为两种模式:一种是NHSTAB (美国高速公路安全管理局) 制定的，一种是SAE International (国际汽车工程师协会) 制定的。

2.背景和相关工作
2.1 软件部分
软件部分主要指模拟驾驶的虚拟幻境。

2.1.1 AirSim
Airsim是用于无人机，汽车等的场景模拟器，基于虚幻引擎，也支持unity。

2.1.2 Udacity
优达学城（Udacity）自动驾驶课程中提供的一个开源、类似Airsim的汽车模拟环境。优达学城（Udacity）拥有一整套完整的教程（https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013）供大家学习使用。

2.2 硬件部分
raspberry pi
国外很多开发者使用树莓派开发无人驾驶模型,有待学习。

3.设计思路
搭建airsim —》安装python库 —》测试库是否安装完整 —》hello_car测试 —》数据探索与准备 —》训练模型 —》测试模型 —》修改完善算法 —》实现自动驾驶模拟

4.环境
软件：

OS：win10 专业版
Unreal Engine 4  4.18.3
Python 3.6.2
OpenCV 4.1.0
Keras 2.0.9
Numpy 1.17.1
TensorFlow 1.14.1

硬件：

CPU: 英特尔 Core i7-6700 @ 3.40GHz 四核
GPU: Nvidia GeForce GTX 1060 3GB
内存: 8 GB

*如果您没有可用的GPU，则可以在Azure（https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.dsvm-deep-learning）上启动深度学习虚拟机，该虚拟机随附所有安装的依赖项和库（如果您使用此虚拟机，请使用提供的py35环境）。

5.制作步骤
5.1搭建环境
5.1.1搭建AirSim场景
引擎：想要使用 Unreal Engine，我们需要下载 Epic 开发的 Epic Games Launcher。然后，从 Epic Games Launcher 中再下载所需要版本的 Unreal Engine。
场景：选择 Epic Games Launcher 中左侧的 Learn，然后在右侧的页面中一直下拉找到 Landscape Mountains点进去。这个场景是官方教程使用的。选择 Create Project，然后选择一个路径存下创建工程时，一定要把工程与位置名称改为英文，默认是中文。下载好场景，Unreal Engine 4这一块的任务就基本完成。

*官方文档特别强调需要使用 4.18 版本，否则可能不成功。低版本自然不行，高版本也可能带来问题。

5.1.2 搭建后端环境
克隆airsim：
这里介绍Visual Studio 2017克隆Github上项目的方法。
打开vs2017菜单里的“团队”，点击管理链接，进入右侧的团队资源管理器，克隆本地GIT存储库，第一行输入 https://github.com/Microsoft/AirSim 
第二行选择存放地址，点击克隆。
克隆完成后，打开vs的命令行，进入AirSim的路径下，输入build.cmd，进行构建。一段时间后，关闭cmd窗口，打开AirSim的克隆路径下\AirSim\AirLib\deps\eigen3\Eigen\src\Core\arch\CUDA\Half.h 这个文件，找到“AS IS”这个引用符号，将它的改为英文引用符号，保存文件。再以同样的方法运行build.cmd，一段时间后，运行成功，airsim后端构建成功。

安装Python相关库：
使用Python 3.5或更高版本安装Anaconda。
1.	安装CNTK或安装Tensorflow（建议在GPU上运行TF）
2.	安装h5py
3.	安装Keras并配置Keras后端以使用TensorFlow（默认）或CNTK。
4.	安装AzCopy。请务必将AzCopy可执行文件的位置添加到系统路径中。
5.	安装其他依赖项。在您的anaconda环境中，以root或管理员身份运行“InstallPackages.py”。

配置CUDA：
CUDA是NVIDIA推出的运算平台，CuDNN是专门针对Deep Learning框架设计的一套GPU计算加速方案。
在安装之前要查询下最新TensorFLow发行版支持到了哪个版本。笔者在安装TensorFLow时，CUDA已经到了10.1版本。另外，也要确认CUDA版本是否支持自己的显卡。基于以上两个条件，笔者选择了CUDA10.0，并下载了对应的CuDNN版本。

*在CPU上运行TensorFlow则跳过此步。
*相关链接如下：
1）显卡型号支持：https://developer.nvidia.com/cuda-gpus
2）CUDA下载地址：https://developer.nvidia.com/cuda-toolkit-archive
3）CuDNN下载地址：https://developer.nvidia.com/rdp/cudnn-download

这里我将用于学习无人驾驶仿真平台的研究与学习记录、将来会不断完善。
参考文献：	
1）https://github.com/microsoft/AirSim	
2）https://github.com/microsoft/AutonomousDrivingCookbook/tree/master/AirSimE2EDeepLearning

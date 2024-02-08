# Games101 课件笔记 

## Computer Graphics is AWESOME!  

### Lecture 1  

> mainly 4 parts
>* rasterization  
>* Curves and Meshes
>* Ray Tracing 
>* Animation / Simulation

*This class is not about API*  

虽然老师承诺每次作业的代码不超过20行，但是吧。。。。要想写出正确的代码，要把这个框架的代码翻好几遍，可能我太菜了吧 T_T  

### Lecture 2

>* 线性代数复习 *为更好描述叉乘引入的矩阵来描述*
>* 判断点是否位于三角形内 *叉乘并判断是否每次结果的符号相同*

### Lecture 3

>transformation  
>* 使用矩阵进行缩放平移旋转 *多个矩阵相乘时注意左乘的顺序，越先的操作，越接近描述物体的矩阵*
>* 将向量增加一个w维度来统一平移操作，同时也能区别向量（0）代表的是点还是方向（1）

### Lecture 4

>* MVP transformation *注意和光线追踪的区别*
>>**建议先看一下Learningopengl的对应章节，理解为什么要做变换**
>>* model:将物体放入世界坐标，并移动到对应位置
>>* view:调整相机位置和角度。
>>这中间应该还有一步clip的裁剪操作，剪去位于视野外的画面
>>* projection:上一步说是相机，但这个相机和你的屏幕一样大，光栅化成像后显示的是物体的正交投影，要产生现实中的透视效果，需要将物体本身进行拉伸（没错就是对于物体，因为这里没有真正模拟光线）。矩阵的计算利用了，近大远小（远平面的画面投射到近平面会缩小），近平面所有xyz坐标不变和远平面中心点的xy坐标不变
>* 绕轴旋转 *Rodrigues’ Rotation Formula（罗德里格旋转公式）*
>>简单来说就是将要旋转的向量分解成垂直，平行于旋转轴的两个向量v1，v2。在旋转过程中v1是不变的。这就变成了向量v2在垂直于旋转轴的平面p转动的2d旋转问题。如果是绕xyz轴的画只需要乘以对应的旋转矩阵就行了于任意轴，将v2叉乘旋转轴得垂直于v2和旋转轴的w向量。以w，v2为基向量就可以表示平面p内任意一个向量。这就可以直接用cos和sin来生成旋转后的v2了。~~至于为什么可以直接乘以cos，sin，我也不大清楚，这应该是一个坐标系映射关系，个人觉得w和v2也是可以用标准的xyz表示，那么向量的对应xyz分量就可以用前面已有的轴旋转阵推得~~  
想看公式得[看这里](https://blog.csdn.net/zsq306650083/article/details/8773996)(ps:老师画得图确实有点抽象。。。)
>* 3D物体的三个轴 pitch roll yaw

### Lecture 5
正式进入光栅化  

*！！像素计算插值或反采样的时候要注意应当使用像素中心值进行计算。即像素的索引x，y个加0.5f！！*

>* Viewport transform(视图变换)：将三维空间物体投影到二维。~~后续看虎书后再补吧~~
>* 硬件的成像原理  
>* 三角形光栅化*此处可以推荐一个github的教学项目[tinyrenderer](https://github.com/ssloy/tinyrenderer)*(PS:虽然我也只写了前几章。。但还是推荐一下。欠的债总要还的 T-T)  
>* bounding box
>* Aliasing(走样)

### Lecture 6

>* 一些信号处理的知识对于走样的解释，简单来说采样率不够  
>* Antialiasing(反走样)
>* Blurred Aliasing(先采样后滤波)  
老师讲的挺好的，不懂得话网上搜搜吧，不懂也问题不大  
>* Filtering: 和卷积差不多
>* MSAA: 增加采样参数，每个点多采样几次（就是多算几次颜色后平均一下）  
- [ ] 以下有空再补  
  
>* FXAA
>* TAA

### Lecture 7

>* 深度缓冲（Z-buffering）(一开始真的没想到这么简单粗暴有效得方法，离散的世界真奇妙。)  

*进入shader*
>* Blinn-Phong Reflectance Model
>>* Specular highlights: 高光
>>* Diffuse reflection： 漫反射
>>* Ambient lighting： 环境光
>>* 一些经验公式

### Lecture 8
>* 图元是由顶点组成的。一个顶点，一条线段，一个三角形或者多边形都可以成为图元。
>* 片元是在图元经过光栅化阶段（这个阶段比较复杂，这里不赘述）后，被分割成一个个像素大小的基本单位。片元其实已经很接近像素了，但是它还不是像素。片元包含了比RGBA更多的信息，比如可能有深度值，法线，纹理坐标等等信息。片元需要在通过一些测试（如深度测试）后才会最终成为像素。可能会有多个片元竞争同一个像素，而这些测试会最终筛选出一个合适的片元，丢弃法线和纹理坐标等不需要的信息后，成为像素。
>* 呈现在屏幕上的包含RGBA值的图像最小单元就是像素了
>* [来源](https://www.jianshu.com/p/e0c7c64bac22)

[一些概念：顶点、 图元、片元、像素](https://blog.csdn.net/u014800094/article/details/53035889)
>* 半程向量的概念,L = La + Ld + Ls= ka Ia + kd (I/r2) max(0, n · l) + ks (I/r2) max(0, n · h)^p
>* 三种着色方法
>>* Flat shading:根据三角形法线着色
>>* Gouraud shading：顶点的法线算出顶点的颜色，像素的颜色：顶点的颜色插值得到
>>* Phong shading： 片元着色,像素的法线用顶点法线插值得出，像素的颜色：每个像素单独法线计算颜色  
[Gouraud shading与Phong shading的区别](https://zhuanlan.zhihu.com/p/411933220?utm_id=0)(大佬还是多，笔记做的比我详细多了)
>* 图形渲染管线
>* 纹理的概念

### Lecture 9
*Texture*

>* Barycentric Coordinates(重心坐标)
>* 纹理的采样
>>* 选取邻近的纹理
>>* 二重线性插值:  
lerp(x, v0, v1) = v0 + x(v1 - v0)  
u0 = lerp(s, u00, u10)  
u1 = lerp(s, u01, u11)
>>* 三重线性插值
>* mipmap的提出（当一个像素点压缩了过多的纹理信息会发生走样）
>>* 各向异性

### Lecture 10

>* 纹理的应用
>>凹凸贴图

* Geometry
>Implicit Surface  
>Explicit Surface  
> Boolean operations  
> [SDF](https://zhuanlan.zhihu.com/p/536530019?utm_id=0)  
> Level Set Methods  
> Fractals  (分形)

### Lecture 11  

>* Object File (.obj) Format
>* Bézier Curves
>> de Casteljau Algorithm  
>> Convex Hull  
>>[ B-spline](https://zhuanlan.zhihu.com/p/260724041?utm_id=0)


### Lecture 12

* Mesh subdivision 
>* Extraordinary vertex   
* Mesh simplification 
>* Quadric Error Metric
* Mesh regularization

*Shadow mapping*

>* Hard shadows

### Lecture 13  

*Ray Tracing*


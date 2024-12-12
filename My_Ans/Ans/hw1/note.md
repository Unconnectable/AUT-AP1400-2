# hw1

> 以下三个是唯一修改的文件 因为我hw1只做了实现矩阵乘法，shell的作业没做

`algebra.h` `algebra.cpp`，分别是`algebra namespace`的定义和实现 

`CmakeList`:修改后的cmake文件





不过`.h和.cpp`存在的问题是 模板函数如果分开的化 需要在`.cpp`文件显式实例化 具体到`template MATRIX<int>`之类的,在`algebra.cpp`的最后有实现

接下来是一些坑和讲解

我全程在WSL实现 未使用`Docker`，以下基于WSL

### **`.h`文件:**

```cpp
#ifndef AUT_AP_2024_Spring_HW1
#define AUT_AP_2024_Spring_HW1


#endif //AUT_AP_2024_Spring_HW1

#include <vector>
#include <random>
#include <fmt/core.h>
//#include <format>这个不能使用
#include <iostream>
#include <stdexcept>
#include <optional>
//以上是基本需要的库
namespace algebra{
    template<typename T>
    using MATRIX = std::vector<std::vector<T>>;

    // Matrix initialization types
    enum class MatrixType { Zeros, Ones, Identity, Random };

    // Function template for matrix initialization
}
```

## **CmakeList:**

-  但是题目要求的`<format>`我找了很久都没有找到,退而求其次选择`fmt`库去实现来代替 C++20 的 `std::format`，它提供类似的功能，并且兼容性较好

- ```sh
  #我在WSL实现的代码
  sudo apt update
  sudo apt install libfmt-dev
  #这时候题目要求的std::format要改成 fmt::format
  ```

- 使用`fmt`后我们需要修改`cmake`文件,接下来我指出修改的地方和上下文 方便定位

- ```cmake
  #CmakeList.txt
  #的都是原文件
  #set(CMAKE_CXX_STANDARD 23)
  #set(CMAKE_CXX_STANDARD_REQUIRED ON)
  
  #find_package(GTest REQUIRED)
  find_package(fmt REQUIRED)#新增
  
  
  #set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  
  target_link_libraries(main
          #GTest::GTest
          #GTest::Main
          fmt::fmt    # Link fmt library 这是新文件
  )
  ```

### **`.cpp`最后的需要加上的东西**

```cpp
namespace algebra{
    template<typename T>
    using MATRIX = std::vector<std::vector<T>>;

    // Matrix initialization types
    enum class MatrixType { Zeros, Ones, Identity, Random };

    // Function template for matrix initialization
    template<typename T>
    MATRIX<T> create_matrix(
        std::size_t rows,
        std::size_t columns,
        std::optional<MatrixType> type = MatrixType::Zeros,
        std::optional<T> lowerBound = std::nullopt,
        std::optional<T> upperBound = std::nullopt);
}
//已经全部实现
template MATRIX<int> create_matrix<int>(std::size_t, std::size_t, std::optional<MatrixType>, std::optional<int>, std::optional<int>);
    template MATRIX<double> create_matrix<double>(std::size_t, std::size_t, std::optional<MatrixType>, std::optional<double>, std::optional<double>);
    template MATRIX<float> create_matrix<float>(std::size_t, std::size_t, std::optional<MatrixType>, std::optional<float>, std::optional<float>);
//否则会报错
```

如果你不想显式实例化，可以直接在`algebra.h`文件补充完整，无需给`algebra.cpp`文件提供接口

```sh
#然后就可以在AUT_AP_2024_Spring_HW1 文件夹下面使用cmake了
mkdir build
cd build
cmake ..
make
./main
#开始快乐的测试吧
#目录文件大概是这样 在build里面测试main文件即可
.
├── Resource
├── bash
│   └── need_backup
│       ├── Archives
│       ├── Documents
│       ├── Images
│       ├── Music
│       ├── NestedFolders
│       │   └── Level1
│       │       └── Level2
│       ├── Scripts
│       └── Videos
├── build
│   └── CMakeFiles
│       ├── 3.22.1
│       │   ├── CompilerIdC
│       │   │   └── tmp
│       │   └── CompilerIdCXX
│       │       └── tmp
│       ├── CMakeTmp
│       └── main.dir
│           └── src
├── include
└── src
```

### ***具体函数的实现和问题 在`algebra.cpp`都有详细的实现 自行查找错误***




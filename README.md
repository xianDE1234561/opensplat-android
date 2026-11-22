# OpenSplat Android

这个仓库记录移动端 3D Gaussian Splatting（3DGS）项目从 `mobile3dgs` 桌面原型到 OpenSplat Android 真机采集与端侧训练的迭代。当前关注两个问题：怎样在较短的训练预算内得到可用的新视角画质，以及怎样用少量训练帧补上重建缺口。

## 当前进度

- **安卓版离线训练已验证。** 在 Android 15 模拟器上，现有 APK 可以导入数据、完成 Brush 训练并导出 PLY 和测试报告。质量对照采用相同相机和参考图统一评分。
- **保守换帧是当前的补缺口候选。** 两个 TUM RGB-D 室内序列都观察到局部覆盖改善。训练预算为 2000 步，仅替换少量训练图，并检查已有视角的画质回退。
- **正在准备真机数据复测。** 2026 年 11 月 20 日的实拍测试版已编译并通过安装检查。下一项验收是用目标安卓手机采集室内场景，导出完整 ZIP，并用同一份素材做配对实验。

[迭代路线、实验结果和验收项](docs/iteration-roadmap.md)

## 源码与历史

安卓工程位于 [`opensplat-android/`](opensplat-android/)，基线为 [OpenSplat Android `1ea9bbb`](https://github.com/SuleMareVientu/opensplat-android/commit/1ea9bbb829d0a4d4a5e4783d3fec1eeb71834d90)。根目录保留早期 Python 原型与[历史 README](README.legacy.md)。

## 构建与使用 Android App

使用 JDK 21、Android SDK 36 和 Android NDK，在 `opensplat-android/` 下运行：

```bash
bash ./gradlew :samples:android-demo:assembleRelease -PtargetAbi=arm64-v8a
```

Windows 用 `gradlew.bat`；模拟器用 `-PtargetAbi=x86_64`。发布构建含 R8 压缩，输出位于 `samples/android-demo/build/outputs/apk/release/`。仓库不包含 APK、模型权重或采集数据；App 使用仓库内已构建的 Brush AAR。其改动源码见 [Brush Android 补丁](patches/brush-android-release.patch)（基于 [`dddf54a`](https://github.com/SuleMareVientu/brush-android/commit/dddf54a102f72f56a21aebce6d102c739135d3cf)）；x86_64 模拟器还有 [GFXStream 的 wgpu 修正](patches/0002-wgpu-gfxstream-timeline-values.patch)。

- **Direct · PocketGS**：从已对齐的彩色 SfM 点云初始化表面对齐高斯，不包含论文的稠密 MVS。无需模型包。
- **LocusGS**：从四张关键帧生成初始高斯；使用独立的 448×256 FP32 ONNX ZIP，生成步骤见下文。
- 两条路线均可用本地 Brush 精修，导出 PLY 与性能报告，并在 Viewer 查看。

### 生成 LocusGS 模型包

在仓库根目录执行。使用 [LocusGS](https://github.com/leo-frank/LocusGS) 提供的 checkpoint 与依赖，并安装 ONNX：

```bash
git clone https://github.com/leo-frank/LocusGS.git vendor/LocusGS
git -C vendor/LocusGS checkout 6056369544443a4673568828b705118202df94fa
git -C vendor/LocusGS apply ../../patches/locusgs-export.patch
python android_infer/export_blocks.py --checkpoint model.safetensors --out build/locusgs --img-size 256,448 --num-gs-tokens 4096
python android_infer/export_init_state.py --checkpoint model.safetensors --out build/locusgs --img-size 256,448 --num-gs-tokens 4096
python android_infer/chunk_attention.py --src build/locusgs --out build/locusgs-chunked --chunk 256
python scripts/package_locusgs_model.py --blocks-dir build/locusgs-chunked --init-state build/locusgs --out build/locusgs-448-chunked.zip
```

在 App 的 **Reconstruction → LocusGS → Install LocusGS model ZIP** 选择此文件。

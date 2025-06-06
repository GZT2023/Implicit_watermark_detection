import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import argparse
from datetime import datetime
import re
import shutil
import hashlib
import platform

# ==============================
# 配置参数 (可根据需要修改)
# ==============================
DEFAULT_INPUT_DIR = "input_images"
DEFAULT_OUTPUT_DIR = "spectrum_results"
REPORT_FILE = "spectrum_analysis_report.csv"
DEBUG_MODE = True  # 设为True显示详细处理信息
SAVE_RAW_SPECTRUM = True  # 是否保存原始频谱数据(.npy文件)
WATERMARK_DETECTION = True  # 是否启用水印检测
WATERMARK_THRESHOLD = 0.85  # 水印检测置信度阈值 (0-1)
AUTO_RENAME_FILES = True  # 是否自动重命名中文文件

# ==============================
# 水印检测参数 (高级用户可调整)
# ==============================
CROSS_INTENSITY_THRESH = 0.15  # 十字线强度阈值
RADIAL_VARIANCE_THRESH = 0.2   # 放射线方差阈值
CENTER_INTENSITY_THRESH = 0.3  # 中心点强度阈值

def sanitize_filename(filename):
    """将中文文件名转换为安全英文文件名"""
    try:
        # 生成文件内容的哈希值作为唯一标识
        with open(filename, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        
        # 获取文件扩展名
        _, ext = os.path.splitext(filename)
        
        # 创建新文件名 (哈希值+原文件名拼音首字母)
        base_name = os.path.basename(filename)
        safe_name = re.sub(r'[^\w\s-]', '', base_name)  # 移除非字母数字字符
        safe_name = re.sub(r'\s+', '_', safe_name)  # 替换空格为下划线
        
        # 截断过长的文件名
        if len(safe_name) > 50:
            safe_name = safe_name[:20] + "..." + safe_name[-20:]
        
        return f"{file_hash}_{safe_name}{ext}"
    except Exception as e:
        if DEBUG_MODE:
            print(f"文件名清理错误: {str(e)}")
        # 如果出错，返回基于时间戳的随机文件名
        return f"file_{datetime.now().strftime('%H%M%S%f')}{ext}"

def copy_and_rename_files(input_dir, temp_dir):
    """复制并重命名输入目录中的所有文件到临时目录"""
    os.makedirs(temp_dir, exist_ok=True)
    file_mapping = {}
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
    
    for filename in os.listdir(input_dir):
        src_path = os.path.join(input_dir, filename)
        if os.path.isfile(src_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            # 生成安全文件名
            safe_name = sanitize_filename(src_path)
            dest_path = os.path.join(temp_dir, safe_name)
            
            # 复制文件
            shutil.copy2(src_path, dest_path)
            file_mapping[src_path] = (dest_path, filename)
    
    return file_mapping

def detect_watermark(spectrum):
    """
    检测频谱中的水印特征
    返回: (is_watermark, confidence, features)
    """
    try:
        height, width = spectrum.shape
        center_y, center_x = height // 2, width // 2
        
        # 特征1: 中心点强度
        center_value = spectrum[center_y, center_x]
        max_value = np.max(spectrum)
        center_intensity = center_value / max_value if max_value > 0 else 0
        
        # 特征2: 十字线检测
        horizontal_line = spectrum[center_y, max(0, center_x-10):min(width, center_x+11)]
        vertical_line = spectrum[max(0, center_y-10):min(height, center_y+11), center_x]
        
        # 计算十字线的平均强度
        h_line_mean = np.mean(horizontal_line) / max_value
        v_line_mean = np.mean(vertical_line) / max_value
        cross_intensity = (h_line_mean + v_line_mean) / 2
        
        # 特征3: 放射状对称性
        y, x = np.indices(spectrum.shape)
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = r > 10  # 排除中心区域
        angular_variance = np.var(spectrum[mask])
        
        # 特征4: 对称性检测 (左上象限与右下象限)
        quadrant_size = min(center_y, center_x) // 2
        q1 = spectrum[center_y-quadrant_size:center_y, center_x:center_x+quadrant_size]
        q3 = spectrum[center_y:center_y+quadrant_size, center_x-quadrant_size:center_x]
        symmetry_diff = np.mean(np.abs(q1 - np.rot90(q3, 2)))
        
        # 特征评分
        features = {
            'center_intensity': center_intensity,
            'cross_intensity': cross_intensity,
            'angular_variance': angular_variance,
            'symmetry_diff': symmetry_diff
        }
        
        # 水印置信度计算 (加权评分)
        confidence = (
            0.4 * min(center_intensity / CENTER_INTENSITY_THRESH, 1.0) +
            0.4 * min(cross_intensity / CROSS_INTENSITY_THRESH, 1.0) +
            0.1 * (1 - min(angular_variance / RADIAL_VARIANCE_THRESH, 1.0)) +
            0.1 * (1 - min(symmetry_diff / 0.1, 1.0))
        )
        confidence = min(max(confidence, 0.0), 1.0)
        
        is_watermark = confidence > WATERMARK_THRESHOLD
        
        return is_watermark, confidence, features
    
    except Exception as e:
        if DEBUG_MODE:
            print(f"水印检测错误: {str(e)}")
        return False, 0.0, {}

def process_image(image_path, original_name, output_dir, report_writer=None, save_raw=False):
    """处理单张图片并保存结果"""
    try:
        # 读取图片 - 使用更健壮的方法
        try:
            # 方法1: 直接读取
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        except:
            # 方法2: 备用读取方法
            with open(image_path, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"无法读取图片: {image_path}")
            return None, "读取失败", False, 0.0
        
        # 获取文件信息
        filename = os.path.basename(original_name)
        safe_name = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        
        # 傅里叶变换
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        
        # 更精确的频谱计算
        magnitude_spectrum = np.log(1 + np.abs(fshift))
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        
        # 标准化频谱用于分析 (0-1范围)
        normalized_spectrum = magnitude_spectrum / 255.0
        
        # 水印检测
        watermark_status = "未检测"
        confidence = 0.0
        features = {}
        is_watermark = False
        
        if WATERMARK_DETECTION:
            is_watermark, confidence, features = detect_watermark(normalized_spectrum)
            watermark_status = "检测到" if is_watermark else "未检测"
        
        # 创建可视化结果
        plt.figure(figsize=(14, 7))
        
        # 原始图像
        plt.subplot(131)
        plt.imshow(img, cmap='gray')
        plt.title(f'原始图像\n{filename}')
        plt.axis('off')
        
        # 频谱图 (灰度)
        plt.subplot(132)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('频谱图 (灰度)')
        plt.axis('off')
        
        # 频谱图 (彩色增强)
        plt.subplot(133)
        plt.imshow(magnitude_spectrum, cmap='jet')
        plt.title('频谱图 (彩色增强)')
        plt.axis('off')
        
        # 添加水印检测结果
        if WATERMARK_DETECTION:
            plt.figtext(0.5, 0.01, 
                        f"水印状态: {watermark_status} | 置信度: {confidence:.2f}",
                        ha="center", fontsize=12, 
                        color="red" if confidence > WATERMARK_THRESHOLD else "green",
                        bbox={"facecolor":"yellow", "alpha":0.3, "pad":5})
        
        # 保存结果图像 - 使用原始文件名
        output_img_path = os.path.join(output_dir, f"{name}_spectrum{ext}")
        plt.savefig(output_img_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # 保存原始频谱数据 (可选)
        if save_raw:
            raw_data_path = os.path.join(output_dir, f"{name}_spectrum.npy")
            np.save(raw_data_path, magnitude_spectrum)
        
        # 记录到报告 - 使用原始文件名
        if report_writer:
            img_size = f"{img.shape[1]}x{img.shape[0]}"
            report_data = [
                filename,
                img_size,
                watermark_status,
                f"{confidence:.4f}",
                f"{features.get('center_intensity', 0):.4f}",
                f"{features.get('cross_intensity', 0):.4f}",
                f"{features.get('angular_variance', 0):.4f}",
                f"{features.get('symmetry_diff', 0):.4f}",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
            report_writer.writerow(report_data)
        
        return output_img_path, "成功", is_watermark, confidence
    
    except Exception as e:
        error_msg = f"处理错误: {str(e)}"
        if DEBUG_MODE:
            print(error_msg)
        return None, error_msg, False, 0.0

def batch_process_images(input_dir, output_dir):
    """批量处理文件夹内所有图片并分类水印"""
    # 确保输出目录存在 (使用相对路径)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建分类目录 (使用相对路径)
    watermarked_dir = os.path.join(output_dir, "watermarked_images")
    clean_dir = os.path.join(output_dir, "clean_images")
    
    # 确保目录存在
    os.makedirs(watermarked_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    
    # 创建临时目录用于处理中文文件名
    temp_dir = os.path.join(output_dir, "temp_processing")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 复制并重命名文件
    file_mapping = copy_and_rename_files(input_dir, temp_dir)
    
    if not file_mapping:
        print("未找到图片文件!")
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 0, []
    
    print(f"找到 {len(file_mapping)} 张图片，开始处理...")
    
    # 创建报告文件
    report_path = os.path.join(output_dir, REPORT_FILE)
    with open(report_path, 'w', newline='', encoding='utf-8') as report_file:
        fieldnames = [
            '文件名', '尺寸', '水印状态', '置信度', 
            '中心强度', '十字线强度', '放射方差', '对称差异', '处理时间'
        ]
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        
        csv_writer = csv.writer(report_file)
        
        # 使用进度条处理每张图片
        success_count = 0
        results = []
        watermark_count = 0
        
        for original_path, (temp_path, original_name) in tqdm(file_mapping.items(), desc="处理图片"):
            # 处理图片并获取结果
            result_path, status, is_watermark, confidence = process_image(
                temp_path, 
                original_name, 
                output_dir, 
                report_writer=csv_writer,
                save_raw=SAVE_RAW_SPECTRUM
            )
            
            if status == "成功":
                success_count += 1
                
                # 确定目标目录
                if is_watermark:
                    target_dir = watermarked_dir
                    watermark_count += 1
                else:
                    target_dir = clean_dir
                
                # 复制原始图片到分类目录（使用原始文件名）
                target_path = os.path.join(target_dir, original_name)
                
                # 确保目标目录存在
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # 使用临时文件作为源文件（英文路径）
                try:
                    shutil.copy2(temp_path, target_path)
                    if DEBUG_MODE:
                        print(f"已复制: {temp_path} -> {target_path}")
                except Exception as e:
                    print(f"复制文件出错: {e}")
                    # 尝试从原始路径复制作为备选方案
                    try:
                        shutil.copy2(original_path, target_path)
                        if DEBUG_MODE:
                            print(f"使用原始路径复制: {original_path} -> {target_path}")
                    except Exception as e2:
                        print(f"复制文件再次出错: {e2}")
                        status = f"复制失败: {e2}"
                
                results.append((original_name, status, result_path, is_watermark, confidence))
            else:
                results.append((original_name, status, "", False, 0.0))
    
    # 清理临时目录
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\n处理完成! 成功处理 {success_count}/{len(file_mapping)} 张图片")
    print(f"检测到水印图片: {watermark_count} 张")
    print(f"无水印图片: {success_count - watermark_count} 张")
    print(f"结果保存在: {output_dir}")
    
    # 验证输出目录内容
    print("\n输出目录内容:")
    print(f"频谱分析图: {len([f for f in os.listdir(output_dir) if f.endswith('_spectrum.')])} 张")
    print(f"水印图片目录: {watermarked_dir} (包含 {len(os.listdir(watermarked_dir))} 个文件)")
    print(f"干净图片目录: {clean_dir} (包含 {len(os.listdir(clean_dir))} 个文件)")
    
    return success_count, results

def generate_summary_report(output_dir):
    """生成处理摘要报告"""
    report_path = os.path.join(output_dir, REPORT_FILE)
    if not os.path.exists(report_path):
        print("未找到分析报告!")
        return
    
    watermark_count = 0
    total_images = 0
    confidences = []
    
    with open(report_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            total_images += 1
            if row['水印状态'] == '检测到':
                watermark_count += 1
                confidences.append(float(row['置信度']))
    
    if total_images == 0:
        print("报告中无有效数据!")
        return
    
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # 获取分类目录中的文件数量
    watermarked_dir = os.path.join(output_dir, "watermarked_images")
    clean_dir = os.path.join(output_dir, "clean_images")
    
    try:
        watermarked_files = len(os.listdir(watermarked_dir))
    except:
        watermarked_files = 0
        
    try:
        clean_files = len(os.listdir(clean_dir))
    except:
        clean_files = 0
    
    summary = f"""
    =============================
      频谱分析报告摘要
    =============================
    分析时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    处理图片总数: {total_images}
    检测到水印图片: {watermark_count} ({watermark_count/total_images*100:.1f}%)
    无水印图片: {total_images - watermark_count}
    平均水印置信度: {avg_confidence:.2f}
    最高水印置信度: {max(confidences) if confidences else 0:.2f}
    
    水印图片目录: {watermarked_dir} (包含 {watermarked_files} 个文件)
    干净图片目录: {clean_dir} (包含 {clean_files} 个文件)
    详细报告: {report_path}
    输出目录: {output_dir}
    =============================
    """
    
    print(summary)
    
    # 保存摘要到文件
    with open(os.path.join(output_dir, "summary_report.txt"), 'w', encoding='utf-8') as f:
        f.write(summary)
    
    return summary

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='图像频谱分析工具')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_DIR, 
                        help='输入图片目录路径')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='输出结果目录路径')
    parser.add_argument('--no-watermark', action='store_true',
                        help='禁用水印检测功能')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    parser.add_argument('--no-rename', action='store_true',
                        help='禁用自动重命名功能')
    args = parser.parse_args()
    
    # 更新配置
    if args.no_watermark:
        WATERMARK_DETECTION = False
    
    if args.debug:
        DEBUG_MODE = True
    
    if args.no_rename:
        AUTO_RENAME_FILES = False
    
    # 打印系统信息
    print("="*50)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"工作目录: {os.getcwd()}")
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print("="*50)
    
    # 执行批量处理
    success_count, results = batch_process_images(args.input, args.output)
    
    # 生成摘要报告
    if success_count > 0:
        generate_summary_report(args.output)
    
    # 显示水印检测样本
    if WATERMARK_DETECTION and success_count > 0:
        watermark_samples = [r for r in results if r[3]]  # 第4个元素是is_watermark
        if watermark_samples:
            print("\n检测到水印的图片 (前5个):")
            for sample in watermark_samples[:5]:
                print(f"  - {sample[0]} (置信度: {sample[4]:.2f})")
        else:
            print("\n未检测到含有水印的图片")
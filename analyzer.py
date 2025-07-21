import cv2
import numpy as np
import os
# from moviepy.editor import VideoFileClip
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
import time
import subprocess
import traceback

# ==============================================================================
# 功能 1: 从视频中检测并提取关键帧
# ==============================================================================
def find_keyframes(video_path, output_dir, hist_threshold=0.7, motion_threshold=20.0, min_scene_duration_sec=2.0):
    """
    从视频中提取关键帧.

    通过结合颜色直方图差异和光流运动幅度来检测关键帧，并确保关键帧之间有足够的时间间隔。

    :param video_path: 输入视频文件的路径.
    :param output_dir: 保存提取出的关键帧图像的目录.
    :param hist_threshold: 颜色直方图相关性阈值。低于此值表示场景变化较大。范围 (0, 1)，建议 0.6-0.8.
    :param motion_threshold: 运动强度阈值。高于此值表示画面有显著运动。
    :param min_scene_duration_sec: 两个关键帧之间的最小时间间隔（秒），用于防止提取过多相似的帧。
    :return: 包含关键帧信息的列表。
    """
    # --- 1. 初始化和准备 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建关键帧图像输出目录: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    min_frames_between_keyframes = int(fps * min_scene_duration_sec)
    
    print("--- 开始检测关键帧 ---")
    print(f"视频信息: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, FPS: {fps:.2f}")
    print(f"参数设置: 直方图阈值={hist_threshold}, 运动阈值={motion_threshold}, 最小场景时长={min_scene_duration_sec}秒 ({min_frames_between_keyframes}帧)")

    keyframes = []
    last_keyframe_number = -min_frames_between_keyframes  # 确保第一帧可以被选为关键帧

    prev_gray = None
    prev_hist = None
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 2. 预处理帧 ---
        # 为了提高性能，可以将帧缩小进行计算
        h, w, _ = frame.shape
        # 为避免尺寸过小，设置一个最小尺寸
        target_w = max(160, w // 4)
        target_h = max(120, h // 4)
        resized_frame = cv2.resize(frame, (target_w, target_h))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        
        # 计算当前帧的直方图
        current_hist = cv2.calcHist([resized_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(current_hist, current_hist, 0, 1, cv2.NORM_MINMAX)

        is_keyframe = False
        reason = ""

        if frame_number == 0:
            is_keyframe = True
            reason = "视频第一帧"
        else:
            # --- 3. 计算多维度指标 ---
            # a) 场景/色调变化 (颜色直方图)
            hist_similarity = cv2.compareHist(prev_hist, current_hist, cv2.HISTCMP_CORREL)

            # b) 动作/运动强度 (光流)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motion_magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))

            # --- 4. 判断是否为关键帧 ---
            # 检查是否满足冷却时间
            if (frame_number - last_keyframe_number) > min_frames_between_keyframes:
                # 条件1: 场景发生剧烈变化 (直方图相关性低)
                if hist_similarity < hist_threshold:
                    is_keyframe = True
                    reason = f"场景切换 (直方图相似度: {hist_similarity:.2f})"
                # 条件2: 画面中有高强度运动
                elif motion_magnitude > motion_threshold:
                    is_keyframe = True
                    reason = f"高强度动作 (运动幅度: {motion_magnitude:.2f})"

        # --- 5. 保存关键帧 ---
        if is_keyframe:
            timestamp_sec = frame_number / fps
            timestamp_str = f"{int(timestamp_sec // 3600):02d}:{int((timestamp_sec % 3600) // 60):02d}:{int(timestamp_sec % 60):02d}"
            
            keyframe_info = {
                "frame_number": frame_number,
                "timestamp_sec": timestamp_sec,
                "timestamp_str": timestamp_str,
                "reason": reason
            }
            keyframes.append(keyframe_info)
            
            # 保存关键帧图像文件
            filename = os.path.join(output_dir, f"keyframe_{frame_number:06d}_{timestamp_str.replace(':', '-')}.jpg")
            cv2.imwrite(filename, frame)
            
            print(f"✅ 找到关键帧: 帧号 {frame_number} @ {timestamp_str}. 原因: {reason}")
            last_keyframe_number = frame_number

        # --- 6. 更新上一帧的信息以供下次循环使用 (关键逻辑修正) ---
        prev_gray = gray_frame
        prev_hist = current_hist # 修正点：确保上一帧的直方图在每次迭代后都更新

        frame_number += 1
        if frame_number % int(fps * 10) == 0: # 每处理10秒视频打印一次进度
            print(f"... 正在处理: {frame_number / fps:.1f} 秒")
        

    # --- 7. 清理和总结 ---
    cap.release()
    print("\n--- 关键帧提取完成 ---")
    print(f"总共找到 {len(keyframes)} 个关键帧。")
    return keyframes

# ==============================================================================
# 功能 2: 根据关键帧列表裁剪视频
# ==============================================================================
def crop_video_by_keyframes(video_path, keyframes, output_dir):
    """
    根据关键帧列表将视频分割成多个片段。
    优化版本：解决 'NoneType' object has no attribute 'stdout' 错误

    :param video_path: 原始视频文件的路径。
    :param keyframes: find_keyframes函数返回的关键帧信息列表。
    :param output_dir: 保存裁剪后视频片段的目录。
    """
    if not keyframes or len(keyframes) < 2:
        print("关键帧数量不足 (少于2个)，无法进行裁剪。")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建裁剪视频输出目录: {output_dir}")

    print(f"\n--- 开始裁剪视频 ---")
    print(f"将根据 {len(keyframes) - 1} 个关键帧区间进行裁剪...")

    # 获取视频总时长（避免在循环中重复加载）
    with VideoFileClip(video_path) as temp_clip:
        video_duration = temp_clip.duration

    # 提取关键帧的时间戳（秒）作为分割点
    split_points = [kf['timestamp_sec'] for kf in keyframes]

    # 使用FFmpeg直接裁剪（更稳定）
    def ffmpeg_cut(start, end, output_file):
        """使用FFmpeg直接裁剪视频片段（更稳定）"""
        try:
            # 使用关键帧精确裁剪
            cmd = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-ss', str(start),
                '-i', video_path,
                '-to', str(end),
                '-c', 'copy',  # 流复制（无损快速）
                '-avoid_negative_ts', '1',  # 避免负时间戳
                output_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if result.returncode != 0:
                print(f"FFmpeg警告: {result.stderr}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg裁剪失败: {e.stderr}")
            return False
        except Exception as e:
            print(f"未知错误: {str(e)}")
            return False

    # 处理每个片段
    for i in range(len(split_points) - 1):
        start_time = split_points[i]
        end_time = split_points[i+1]
        
        # 确保结束时间不超过视频总时长
        end_time = min(end_time, video_duration)
        
        # 跳过无效片段
        if end_time - start_time < 0.1:
            print(f"跳过片段 {i+1}，时长过短 ({end_time - start_time:.2f}s)。")
            continue
        
        print(f"正在裁剪片段 {i+1}/{len(split_points) - 1}: 从 {start_time:.2f}s 到 {end_time:.2f}s")
        
        # 构造输出文件名
        output_filename = os.path.join(output_dir, f"segment_{i+1:03d}_({start_time:.2f}s_to_{end_time:.2f}s).mp4")
        
        try:
            # 方法1: 使用FFmpeg直接裁剪（推荐）
            success = ffmpeg_cut(start_time, end_time, output_filename)
            
            if not success:
                # 方法2: 回退到MoviePy（如果FFmpeg失败）
                print("FFmpeg裁剪失败，尝试使用MoviePy...")
                with VideoFileClip(video_path) as main_clip:
                    sub_clip = main_clip.subclip(start_time, end_time)
                    # 使用更安全的写入参数
                    sub_clip.write_videofile(
                        output_filename,
                        threads=4,  # 适当线程数
                        preset="ultrafast",  # 更快编码
                        audio_codec="aac",
                        logger=None  # 禁用日志避免冲突
                    )
                    # 确保关闭剪辑
                    sub_clip.close()
            
            # 短暂暂停防止资源冲突
            time.sleep(1)
            
        except Exception as e:
            print(f"裁剪片段 {i+1} 时发生严重错误: {str(e)}")
            traceback.print_exc()
            print(f"跳过片段 {i+1} 继续处理后续片段...")

    print("\n--- 视频裁剪完成 ---")
    print(f"所有片段已保存到: {output_dir}")
# ==============================================================================
# 主执行函数
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 设置路径 ---
    # 替换为你的视频文件路径
    # 注意：如果路径或文件名包含中文，请确保你的系统和Python环境支持UTF-8。
    video_file = "/home/aistudio/result.mp4"  
    # 关键帧图片将保存在这个文件夹
    keyframe_image_folder = "./output/image"
    # 裁剪后的视频片段将保存在这个文件夹
    cropped_clips_folder = "./output/video"

    # --- 2. 设置可调参数 ---
    # 直方图相似度阈值。越低，越容易因为场景变化而触发。建议范围 [0.6, 0.8]
    # 如果你的视频有很多缓慢的镜头渐变，可以适当调低此值。
    HISTOGRAM_THRESHOLD = 0.1
    
    # 运动强度阈值。越高，需要越剧烈的运动才能触发。
    # 对于体育、动作类视频，可以适当提高；对于访谈、风景类，可以降低。
    MOTION_THRESHOLD = 1

    # 最小场景时长（秒）。用于防止在同一个连续动作或场景中提取过多帧。
    # 这是控制关键帧总数量的最有效参数。
    MIN_DURATION_SECONDS = 5
    
    # --- 3. 执行流程 ---
    # 检查视频文件是否存在
    if not os.path.exists(video_file):
        print(f"错误: 视频文件未找到 '{video_file}'")
        print("请在代码中将 'video_file' 变量更改为你的视频文件实际路径。")
    else:
        # 第一步：检测并提取关键帧
        found_keyframes = find_keyframes(
            video_path=video_file,
            output_dir=keyframe_image_folder,
            hist_threshold=HISTOGRAM_THRESHOLD,
            motion_threshold=MOTION_THRESHOLD,
            min_scene_duration_sec=MIN_DURATION_SECONDS
        )

        # 第二步：如果找到了关键帧，则进行视频裁剪
        if found_keyframes:
            crop_video_by_keyframes2(
                video_path=video_file,
                keyframes=found_keyframes,
                output_dir=cropped_clips_folder
            )
        else:
            print("未能提取任何关键帧，程序结束。")
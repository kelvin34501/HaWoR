#!/usr/bin/env python3
import json

def process_segments(input_file, output_file):
    """
    从input_file的第二列读取frame id，
    每4个为一组，取中间两个作为左闭右开区间[start, end)，
    保存为JSON列表
    """
    # 读取frame IDs
    frame_ids = []
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                frame_id = int(parts[1])
                frame_ids.append(frame_id)
    
    # 分组处理：每4个一组，取中间2个
    frame_ranges = []
    for i in range(0, len(frame_ids) - 3, 4):  # 确保至少有4个元素
        group = frame_ids[i:i+4]
        # 取索引1和2（中间两个）作为左闭右开区间
        frame_range = [group[1], group[2]]
        frame_ranges.append(frame_range)
    
    # 保存为JSON
    with open(output_file, 'w') as f:
        json.dump(frame_ranges, f, indent=4)
    
    print(f"处理完成！")
    print(f"总共读取 {len(frame_ids)} 个frame IDs")
    print(f"生成 {len(frame_ranges)} 个frame ranges")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    input_file = "tmp/clip_1_seg.txt"
    output_file = "tmp/clip_1_segment_def.json"
    process_segments(input_file, output_file)

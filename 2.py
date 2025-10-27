import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ==============================================================================
# THIẾT LẬP MEDIAPIPE HOLISTIC
# ==============================================================================
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,            # 1 hoặc 2 = chính xác hơn
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,   # Tắt face vì chỉ cần pose + hand
    min_detection_confidence=0.3,  # Giảm để dễ nhận tay hơn
    min_tracking_confidence=0.6
)

# ==============================================================================
# HÀM TRÍCH XUẤT LANDMARKS
# ==============================================================================
def make_landmark_pose(results):
    """Trả về list [x, y, z, visibility] cho 33 pose landmarks"""
    if not results.pose_landmarks:
        return None
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.extend([lm.x, lm.y, lm.z, lm.visibility])
    return c_lm

def make_landmark_hand_left(results):
    """Trả về list [x, y, z] cho 21 left hand landmarks"""
    if not results.left_hand_landmarks:
        return None
    c_lm_hl = []
    for lm_hl in results.left_hand_landmarks.landmark:              
        c_lm_hl.extend([lm_hl.x, lm_hl.y, lm_hl.z])
    return c_lm_hl

def make_landmark_hand_right(results):
    """Trả về list [x, y, z] cho 21 right hand landmarks"""
    if not results.right_hand_landmarks:
        return None
    c_lm_hr = []
    for lm_hr in results.right_hand_landmarks.landmark:
        c_lm_hr.extend([lm_hr.x, lm_hr.y, lm_hr.z])
    return c_lm_hr

def draw_all_landmarks(mp_drawing, results, img):
    """Vẽ tất cả landmarks lên frame"""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1)
        )
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )
    
    return img

# ==============================================================================
# HÀM XỬ LÝ 1 VIDEO
# ==============================================================================
def process_single_video(video_path, output_csv, show_video=True, save_video=False, output_video_path=None):
    """
    Xử lý 1 video và lưu landmarks ra CSV
    
    Args:
        video_path: Đường dẫn video đầu vào
        output_csv: Đường dẫn file CSV đầu ra
        show_video: Hiển thị video trong quá trình xử lý
        save_video: Lưu video đã vẽ landmarks
        output_video_path: Đường dẫn video đầu ra
    """
    
    print(f"\n{'='*80}")
    print(f"🎥 Đang xử lý: {video_path}")
    print(f"{'='*80}")
    
    # Kiểm tra file tồn tại
    if not os.path.exists(video_path):
        print(f"❌ Lỗi: Không tìm thấy file {video_path}")
        return None
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Lỗi: Không thể mở video")
        return None
    
    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📊 Thông tin video:")
    print(f"   - FPS: {fps}")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - Total frames: {total_frames}")
    print(f"   - Duration: {total_frames/fps:.2f}s")
    
    # Khởi tạo VideoWriter nếu cần lưu video
    out = None
    if save_video and output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"💾 Sẽ lưu video output vào: {output_video_path}")
    
    # Danh sách lưu landmarks
    lm_list = []
    
    frame_count = 0
    successful_pose = 0
    successful_left_hand = 0
    successful_right_hand = 0
    
    print(f"\n🔄 Đang xử lý {total_frames} frames...")
    
    with tqdm(total=total_frames, desc="Progress", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Chuyển BGR sang RGB
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect landmarks
            result = holistic.process(frameRGB)
            
            # Trích xuất landmarks
            pose_lm = make_landmark_pose(result)
            left_hand_lm = make_landmark_hand_left(result)
            right_hand_lm = make_landmark_hand_right(result)
            
            # Chỉ lưu nếu có pose landmarks
            if pose_lm:
                row = []
                # Pose: 33 landmarks × 4 = 132 values
                row.extend(pose_lm)
                # Left hand: 21 landmarks × 3 = 63 values (hoặc zeros)
                row.extend(left_hand_lm if left_hand_lm else [0.0] * 63)
                # Right hand: 21 landmarks × 3 = 63 values (hoặc zeros)
                row.extend(right_hand_lm if right_hand_lm else [0.0] * 63)
                
                lm_list.append(row)
                
                # Đếm số lần detect thành công
                successful_pose += 1
                if left_hand_lm:
                    successful_left_hand += 1
                if right_hand_lm:
                    successful_right_hand += 1
            
            # Vẽ landmarks lên frame
            if show_video or save_video:
                frame = draw_all_landmarks(mp_draw, result, frame)
                
                # Hiển thị thông tin
                cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Pose: {successful_pose}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Left Hand: {successful_left_hand}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right Hand: {successful_right_hand}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Hiển thị video
            if show_video:
                cv2.imshow("Holistic Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️ Người dùng dừng xử lý")
                    break
            
            # Lưu video
            if save_video and out:
                out.write(frame)
            
            pbar.update(1)
    
    # Giải phóng resources
    cap.release()
    if out:
        out.release()
    if show_video:
        cv2.destroyAllWindows()
    
    # Báo cáo kết quả
    print(f"\n✅ Hoàn thành xử lý!")
    print(f"   - Tổng frames: {frame_count}")
    print(f"   - Pose detected: {successful_pose} ({successful_pose/frame_count*100:.1f}%)")
    print(f"   - Left hand detected: {successful_left_hand} ({successful_left_hand/frame_count*100:.1f}%)")
    print(f"   - Right hand detected: {successful_right_hand} ({successful_right_hand/frame_count*100:.1f}%)")
    
    # Lưu CSV
    if len(lm_list) > 0:
        df = pd.DataFrame(lm_list)
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Đã lưu {len(lm_list)} frames vào: {output_csv}")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {df.shape[1]} (132 pose + 63 left hand + 63 right hand)")
    else:
        print(f"\n❌ Không có dữ liệu nào được trích xuất!")
    
    return df

# ==============================================================================
# HÀM XỬ LÝ NHIỀU VIDEO TRONG THƯ MỤC
# ==============================================================================
def process_video_folder(folder_path, output_folder, label_name=None, show_video=False, save_videos=False):
    """
    Xử lý tất cả video trong 1 thư mục
    
    Args:
        folder_path: Đường dẫn thư mục chứa videos
        output_folder: Thư mục lưu CSV
        label_name: Tên nhãn (dùng làm prefix cho file)
        show_video: Hiển thị video trong quá trình xử lý
        save_videos: Lưu video đã vẽ landmarks
    """
    
    print(f"\n{'='*80}")
    print(f"📁 XỬ LÝ THƯ MỤC: {folder_path}")
    print(f"{'='*80}")
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_folder, exist_ok=True)
    
    # Lấy danh sách video
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    video_files = [f for f in os.listdir(folder_path) 
                   if os.path.splitext(f)[1] in video_extensions]
    
    if len(video_files) == 0:
        print(f"❌ Không tìm thấy video nào trong {folder_path}")
        return
    
    print(f"📹 Tìm thấy {len(video_files)} video")
    
    # Xử lý từng video
    all_dataframes = []
    
    for idx, video_file in enumerate(video_files, 1):
        video_path = os.path.join(folder_path, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        # Tạo tên file output
        if label_name:
            csv_filename = f"{label_name}_{idx}.csv"
            video_filename = f"{label_name}_{idx}_annotated.mp4"
        else:
            csv_filename = f"{video_name}.csv"
            video_filename = f"{video_name}_annotated.mp4"
        
        csv_output = os.path.join(output_folder, csv_filename)
        video_output = os.path.join(output_folder, video_filename) if save_videos else None
        
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(video_files)}] {video_file}")
        print(f"{'='*80}")
        
        # Xử lý video
        df = process_single_video(
            video_path,
            csv_output,
            show_video=show_video,
            save_video=save_videos,
            output_video_path=video_output
        )
        
        if df is not None and len(df) > 0:
            all_dataframes.append(df)
        else:
            print(f"❌ Lỗi xử lý video: {video_file}")
    
    # Merge tất cả CSV thành 1 file
    if len(all_dataframes) > 0 and label_name:
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        merged_path = os.path.join(output_folder, f"{label_name}_merged.csv")
        merged_df.to_csv(merged_path, index=False)
        print(f"\n{'='*80}")
        print(f"✅ Đã merge {len(all_dataframes)} files thành: {merged_path}")
        print(f"   - Total rows: {len(merged_df)}")
        print(f"{'='*80}")

# ==============================================================================
# MAIN - SỬ DỤNG
# ==============================================================================
if __name__ == "__main__":
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║           TRÍCH XUẤT HOLISTIC LANDMARKS TỪ VIDEO (POSE + HANDS)          ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # ============== CẤU HÌNH ==============
    
    # OPTION 1: Xử lý 1 video đơn lẻ
    USE_SINGLE_VIDEO = True  # Đổi thành True để xử lý 1 video
    
    if USE_SINGLE_VIDEO:
        VIDEO_PATH = "video/76621-559757958_tiny.mp4"  # ← Đổi đường dẫn video của bạn
        OUTPUT_CSV = "output/normal3.csv"  # ← Đường dẫn lưu CSV
        OUTPUT_VIDEO = "output/normal1_video.mp4"  # ← Video có landmarks
        
        process_single_video(
            video_path=VIDEO_PATH,
            output_csv=OUTPUT_CSV,
            show_video=True,        # Hiển thị video trong khi xử lý
            save_video=False,       # Lưu video có landmarks (tốn thời gian!)
            output_video_path=OUTPUT_VIDEO
        )
    
    # OPTION 2: Xử lý cả thư mục video
    else:
        # Ví dụ: Xử lý thư mục video "warning" (bệnh tim)
        process_video_folder(
            folder_path="cambientim/warning_videos",  # ← Thư mục chứa video warning
            output_folder="cambientim/warning_csv",    # ← Thư mục lưu CSV
            label_name="normal",                       # ← Tên nhãn
            show_video=False,                          # Không hiển thị (xử lý nhanh hơn)
            save_videos=False                          # Không lưu video
        )
        
        # Xử lý thư mục video "normal" (bình thường)
        process_video_folder(
            folder_path="cambientim/normal_videos",
            output_folder="cambientim/normal_csv",
            label_name="normal",
            show_video=False,
            save_videos=False
        )
    
    print(f"\n{'='*80}")
    print("🎉 HOÀN THÀNH TẤT CẢ!")
    print(f"{'='*80}")
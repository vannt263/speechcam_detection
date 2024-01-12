*Trước khi chạy code, hi vọng bạn đã cài đặt đầy đủ các thư viện được liệt kê trong file requirement.txt
*Lưu ý: thay đường link dẫn tới video.

# Tác vụ Detection và Tracking:
- Nhóm chúng tôi sử dụng YOLOv8 với trọng số và tên class được lưu trong thư mục ./code/model

- Để thực nghiệm, các bạn có thể chạy file object tracking trong thư mục
+ ./detection_tracking/base: code dựa trên ý tưởng cơ bản của Detection based Tracking.
+ ./detection_tracking/sort: sử dụng thuật toán SORT.

- Trong tác vụ này, chúng tôi thử nghiệm trên 1 số video được lưu trong thư mục ./data và trả ra kết quả trong thư mục ./output/tracking

# Tác vụ Ước lượng vận tốc xe ô tô:
- Dựa trên kết quả của tracking, nhóm sử dụng thuật toán SORT để theo dõi đối tượng, phục vụ cho nhiệm vụ ước lượng vận tốc.

- Như báo cáo đính kèm, hạn chế của cách làm này là cần xác định trước 1 đoạn đường có sẵn và tọa độ của 2 đường line, nhóm đã đính sẵn một tệp information.txt trong thư mục ./code/speed mô tả khoảng cách và tọa độ của đường line của một số video nhóm thử nghiệm. Khoảng cách được nhóm ước lượng qua quan sát số lượng và độ dài của đường nét đứt chia làn.

- Để thực nghiệm, các bạn có thể chạy file estimate_speed.py

# Tác vụ Trích xuất thông tin ô tô (biển số xe):

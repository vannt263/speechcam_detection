import math

class EuclideanDistTracker:
    def __init__(self):
        # Khởi tạo bộ theo dõi đối tượng
        self.center_points = {}  # Tạo từ điển để lưu trữ tâm của các đối tượng
        self.id_count = 0  # Đếm để gán ID duy nhất cho các đối tượng

    def calculate_center(self, rect):
        # Tính toán tâm của hình chữ nhật
        x, y, w, h = rect
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2
        return cx, cy

    def update(self, objects_rect):
    # Cập nhật theo dõi đối tượng với danh sách các hình chữ nhật mới
        objects_bbs_ids = []  # Danh sách để lưu trữ hình chữ nhật và ID của các đối tượng đã được theo dõi
        new_center_points = {}  # Tạo từ điển mới để lưu trữ tâm của các đối tượng

        # Duyệt qua danh sách các hình chữ nhật đầu vào
        for rect in objects_rect:
            cx, cy = self.calculate_center(rect)

            same_object_detected = False

            # Duyệt qua các đối tượng đã được theo dõi
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                # Kiểm tra xem hình chữ nhật có gần với một đối tượng đã được theo dõi hay không
                if dist < 25:
                    self.center_points[id] = (cx, cy)  # Cập nhật tâm của đối tượng đã được theo dõi
                    objects_bbs_ids.append([rect[0], rect[1], rect[2], rect[3], id])  # Lưu hình chữ nhật và ID của đối tượng
                    same_object_detected = True
                    break

            # Nếu hình chữ nhật không gần với bất kỳ đối tượng nào đã được theo dõi, gán một ID mới
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)  # Lưu tâm của đối tượng mới
                objects_bbs_ids.append([rect[0], rect[1], rect[2], rect[3], self.id_count])  # Lưu hình chữ nhật và ID mới
                self.id_count += 1

        # Loại bỏ các đối tượng không còn được theo dõi khỏi danh sách tâm
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Cập nhật danh sách tâm với các ID không còn được sử dụng
        self.center_points = new_center_points.copy()

        return objects_bbs_ids  # Trả về danh sách hình chữ nhật và ID của các đối tượng đã được theo dõi


# Ví dụ sử dụng:
# tracker = EuclideanDistTracker()
# objects_rectangles = [(x1, y1, width1, height1), (x2, y2, width2, height2)]
# tracked_objects = tracker.update(objects_rectangles)

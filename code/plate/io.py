import tensorflow as tf

class DataLoader:
    def __init__(self, scale=8):
        self.scale = scale

    """
    Load anh từ đĩa và trả về dưới dạng tensor
    """
    def decodeImg(self, path):
        # Đọc dữ liệu từ đường dẫn
        image = tf.io.read_file(path)
        # Giải mã ảnh từ dữ liệu nhị phân
        image = tf.image.decode_jpeg(image, channels=3)
        # Chuyển đổi giá trị pixel về dạng float và chuẩn hóa về khoảng [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Các xử lý khác (bạn có thể bật/tắt theo nhu cầu)
        # image = tf.image.rgb_to_grayscale(image)
        # image = tf.image.adjust_contrast(image, 0.7)
        return image, tf.shape(image)

    """
    Tiền xử lý ảnh để tạo các nhãn
    """
    def processXY(self, path):
        # Giải mã ảnh và lấy kích thước
        y, size = self.decodeImg(path)
        # Thực hiện resize để tạo ảnh đầu vào (x)
        x = tf.image.resize(y, [size[0] // self.scale, size[1] // self.scale])
        # Áp dụng các biến đổi ngẫu nhiên (ở đây là độ sáng và độ tương phản)
        x = tf.image.random_brightness(x, 0.3)
        x = tf.image.random_contrast(x, 0.5, 2)
        return x, y

    def load(self, fileList, batchSize=24):
        # Tạo Dataset từ danh sách đường dẫn ảnh
        fileList = tf.data.Dataset.from_tensor_slices(fileList)
        # Áp dụng hàm processXY cho từng ảnh song song
        data = fileList.map(self.processXY, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Nhóm ảnh lại thành các batch và lặp vô hạn
        data = data.batch(batchSize).repeat()
        # Tăng cường hiệu suất bằng cách prefetch các batch
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return data

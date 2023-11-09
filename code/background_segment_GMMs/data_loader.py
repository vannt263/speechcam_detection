from abc import ABC, abstractmethod

# Định nghĩa lớp trừu tượng, bắt buộc lớp con phải triển khai phương thức get_next_frame
class DataLoader(ABC):
    @abstractmethod
    def get_next_frame(self):
        pass

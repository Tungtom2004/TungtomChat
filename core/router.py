# core/router.py
import re
import unicodedata
import numpy as np

from core.loaders import load_embedder

# ===============================
# CONFIG
# ===============================
ROUTE_MIN_COS = float(
    __import__("os").getenv("ROUTE_MIN_COS", "0.30")
)

# ===============================
# TEXT NORMALIZE (giữ nguyên)
# ===============================
def vn_normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()

# ===============================
# ROUTE SAMPLES (copy từ app.py)
# ===============================
ROUTE_SAMPLES = {
    "hedieuhanh": [
        "Hệ điều hành là gì?",
        "Môn hệ điều hành có mấy chương?",
        "Bạn có thể giải thích về deadlock trong hệ điều hành không?",
        "Sự khác biệt giữa tiến trình (process) và luồng (thread) là gì?",
        "Bộ nhớ ảo (virtual memory) hoạt động như thế nào?",
        "Các giải thuật lập lịch CPU phổ biến là gì?",
        "So sánh FCFS và Round Robin.",
        "Thông tin về môn học Hệ điều hành.",
        "Kernel là gì?",
        "Hệ thống file (file system) là gì?",
        "Thông tin về môn Hệ điều hành PTIT?",
    ],
    "Phantichthietkehttt": [
        "Môn phân tích thiết kế hệ thống thông tin có mấy chương?",
        "Biểu đồ Use Case (Use Case Diagram) dùng để làm gì?",
        "Hãy giải thích về biểu đồ lớp (Class Diagram).",
        "Sự khác biệt giữa yêu cầu chức năng và yêu cầu phi chức năng là gì?",
        "UML là gì và nó được sử dụng như thế nào?",
        "So sánh các mối quan hệ: Association, Aggregation, và Composition.",
        "Hãy kể tên một vài mô hình phát triển phần mềm.",
        "Mô hình Agile là gì?",
        "Mô hình thác nước (Waterfall) hoạt động ra sao?",
        "DAO (Data Access Object) là gì?",
        "Thông tin về môn Phân tích thiết kế hệ thống thông tin."
    ],
    "tutuonghcm": [
        "Môn tư tưởng Hồ Chí Minh có mấy chương?",
        "Nguồn gốc của Tư tưởng Hồ Chí Minh là gì?",
        "Nội dung cốt lõi của Tư tưởng Hồ Chí Minh về chủ nghĩa xã hội là gì?",
        "Tư tưởng Hồ Chí Minh về con đường cách mạng giải phóng dân tộc là gì?",
        "Vai trò của Đảng Cộng sản Việt Nam theo tư tưởng Hồ Chí Minh?",
        "Bạn có thể tóm tắt các đặc trưng của chủ nghĩa xã hội ở Việt Nam không?",
        "Chủ nghĩa Mác-Lênin có vai trò gì trong Tư tưởng Hồ Chí Minh?",
        "Thông tin về môn Tư tưởng Hồ Chí Minh.",
        "Sự kiện lịch sử 17h30 ngày 7/5/1954 là gì?",
        "Cô Phạm Thị Khánh là ai?"
    ],
    "Xulyanh": [
        "Mô hình màu là gì?",
        "Môn xử lý ảnh có mấy chương?",
        "Xử lý ảnh (image processing) là gì?",
        "Phép lọc Gaussian (Gaussian filter) được sử dụng để làm gì?",
        "Phân đoạn ảnh (image segmentation) là gì?",
        "Biến đổi Fourier trong xử lý ảnh có ý nghĩa gì?",
        "So sánh phép toán hình thái học Erosion (co) và Dilation (giãn).",
        "Thresholding (ngưỡng hóa) trong xử lý ảnh là gì?",
        "Làm thế nào để phát hiện biên (edge detection) trong một bức ảnh?",
        "Bộ lọc Median (Median filter) khác gì bộ lọc trung bình (Mean filter)?",
        "Thông tin về môn Xử lý ảnh PTIT"
    ],
    "chitchat": [
        "Thời tiết hôm nay như thế nào?",
        "Ngoài trời nóng bao nhiêu?",
        "Ngày mai có mưa không?",
        "Nhiệt độ hiện tại là bao nhiêu?",
        "Bạn có thể cho tôi biết điều kiện thời tiết hiện tại không?",
        "Cuối tuần này có nắng không?",
        "Nhiệt độ hôm qua là bao nhiêu?",
        "Đêm nay trời sẽ lạnh đến mức nào?",
        "Ai là tổng thống đầu tiên của Hoa Kỳ?",
        "Chiến tranh thế giới thứ hai kết thúc vào năm nào?",
        "Bạn có thể kể cho tôi về lịch sử của internet không?",
        "Tháp Eiffel được xây dựng vào năm nào?",
        "Ai đã phát minh ra điện thoại?",
        "Tên của bạn là gì?",
        "Bạn có tên không?",
        "Tôi nên gọi bạn là gì?",
        "Ai đã tạo ra bạn?",
        "Bạn bao nhiêu tuổi?",
        "Bạn có thể kể cho tôi một sự thật thú vị không?",
        "Bạn có biết bất kỳ câu đố thú vị nào không?",
        "Màu sắc yêu thích của bạn là gì?",
        "Bộ phim yêu thích của bạn là gì?",
        "Bạn có sở thích nào không?",
        "Ý nghĩa của cuộc sống là gì?",
        "Bạn có thể kể cho tôi một câu chuyện cười không?",
        "Thủ đô của Pháp là gì?",
        "Dân số thế giới là bao nhiêu?",
        "Có bao nhiêu châu lục?",
        "Ai đã viết 'Giết con chim nhại'?",
        "Bạn có thể cho tôi một câu nói của Albert Einstein không?",
        "Tóm tắt nội dung phim Mưa Đỏ của đạo diễn Đặng Thái Huyền?",
        "Anh Tạ là ai?",
        "MH370 là cái gì?",
        "Nhỡ thích a Quang lính VNCH thì sao?",
        "Cô Đào Thị Thúy Quỳnh là ai?",
        "Thầy Đặng Hoàng Long là ai?",
        "Thầy Nguyễn Mạnh Hùng là ai?",
        "Cô Đỗ Thị Bích Ngọc là ai?",
        "Thầy Phạm Văn Cường là ai?",
        "Tổng thống Nga là ai?"
    ]
}

# ===============================
# VECTOR NORMALIZE
# ===============================
def _l2_norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + 1e-12)

# ===============================
# BUILD ROUTE INDEX (load once)
# ===============================
def build_route_index():
    embedder = load_embedder()
    idx = {}
    for route, samples in ROUTE_SAMPLES.items():
        vecs = [embedder.embed_query(s) for s in samples]
        mat = _l2_norm(np.array(vecs, dtype=float))
        idx[route] = mat
    return idx

ROUTE_INDEX = build_route_index()

# ===============================
# SEMANTIC ROUTING
# ===============================
def route_semantic(query: str):
    q = (query or "").strip()
    if not q:
        return "unknown", 0.0

    embedder = load_embedder()
    qv = np.array(embedder.embed_query(q), dtype=float)
    qv = qv / (np.linalg.norm(qv) + 1e-12)

    best, score = "unknown", -1.0
    for name, mat in ROUTE_INDEX.items():
        if mat.size == 0:
            continue
        s = float(mat.dot(qv).max())
        if s > score:
            best, score = name, s

    if score < ROUTE_MIN_COS:
        return "unknown", score

    return best, score

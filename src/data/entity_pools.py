"""Curated entity pools to break LLM entity mode-collapse during generation.

Empirically (2026-06-01, mimo-v2.5-pro) raising temperature does NOT fix entity
collapse: the API caps temperature at 1.5, and above ~1.1 the JSON output degrades
to empty/truncated, while within the stable range the popularity prior still wins
(a free "name any Vpop song" returns "Nơi Này Có Anh" even at temp 1.5). The only
reliable lever is to SUPPLY concrete slots. build_entity_hint(domains) samples a
rotating subset into each prompt; the model is told to use a different slot per
sample and may substitute other real ones, so breadth exceeds the pool itself.
"""
import random

# (title, artist) — deliberately spread across many artists, not just the top hits.
SONGS = [
    ("Chúng Ta Của Hiện Tại", "Sơn Tùng M-TP"), ("Lạc Trôi", "Sơn Tùng M-TP"),
    ("Em Của Ngày Hôm Qua", "Sơn Tùng M-TP"), ("Chạy Ngay Đi", "Sơn Tùng M-TP"),
    ("See Tình", "Hoàng Thùy Linh"), ("Để Mị Nói Cho Mà Nghe", "Hoàng Thùy Linh"),
    ("Kẻ Cắp Gặp Bà Già", "Hoàng Thùy Linh"), ("Bùa Yêu", "Bích Phương"),
    ("Đi Đu Đưa Đi", "Bích Phương"), ("Bao Giờ Lấy Chồng", "Bích Phương"),
    ("Em Gì Ơi", "Jack"), ("Sóng Gió", "Jack"), ("Hoa Hải Đường", "Jack"),
    ("Đường Tôi Chở Em Về", "Bùi Trường Linh"), ("Nàng Thơ", "Hoàng Dũng"),
    ("Đoạn Kết Mới", "Hoàng Dũng"), ("Trên Tình Bạn Dưới Tình Yêu", "Min"),
    ("Có Em Chờ", "Min"), ("Một Nhà", "Da LAB"), ("Thanh Xuân", "Da LAB"),
    ("Ngày Đầu Tiên", "Đức Phúc"), ("Hơn Cả Yêu", "Đức Phúc"),
    ("Thật Bất Ngờ", "Trúc Nhân"), ("Lớn Rồi Còn Khóc Nhè", "Trúc Nhân"),
    ("Thằng Điên", "JustaTee"), ("Đông Kiếm Em", "Vũ"), ("Lạ Lùng", "Vũ"),
    ("Mùa Mưa Ngâu Nằm Cạnh", "Vũ"), ("Đi Theo Bóng Mặt Trời", "Đen Vâu"),
    ("Bài Này Chill Phết", "Đen Vâu"), ("Mang Tiền Về Cho Mẹ", "Đen Vâu"),
    ("Trốn Tìm", "Đen Vâu"), ("Lối Nhỏ", "Đen Vâu"),
    ("Hoa Nở Không Màu", "Hoài Lâm"), ("Buồn Làm Chi Em Ơi", "Hoài Lâm"),
    ("Có Chàng Trai Viết Lên Cây", "Phan Mạnh Quỳnh"), ("Vợ Người Ta", "Phan Mạnh Quỳnh"),
    ("NF (Người Yêu Cũ)", "Phan Mạnh Quỳnh"), ("Tình Đơn Phương", "Lê Bảo Bình"),
    ("Gặp Nhưng Không Ở Lại", "Lê Bảo Bình"), ("Phía Sau Một Cô Gái", "Soobin Hoàng Sơn"),
    ("Đi Để Trở Về", "Soobin Hoàng Sơn"), ("Tháng Tư Là Lời Nói Dối Của Em", "Hà Anh Tuấn"),
    ("Tái Sinh", "Tùng Dương"), ("Waiting For You", "MONO"), ("Em Xinh", "MONO"),
    ("3107", "W/n"), ("Anh Nhà Ở Đâu Thế", "AMEE"), ("Sao Anh Chưa Về Nhà", "AMEE"),
    ("Yêu Thầm", "AMEE"), ("Ghen", "Erik"), ("Sau Tất Cả", "Erik"),
    ("Chân Ái", "Orange"), ("Người Lạ Ơi", "Karik"),
]

# first names + relationship terms (how people are addressed in contacts)
NAMES = [
    "Hùng", "Lan", "Tuấn", "Hà", "Minh", "Mai", "Linh", "Nam", "Hằng", "Tùng",
    "Quân", "Thảo", "Dũng", "Trang", "Phong", "Ngọc", "Sơn", "Yến", "Đạt", "Vy",
    "Khoa", "Như", "Bình", "Châu", "Huy", "Trâm", "Long", "Diệu", "Kiên", "Oanh",
    "Thắng", "Hương", "Tài", "Loan", "Cường", "Bích", "Hải", "Vân", "Phúc", "Duy",
    "anh Thắng", "chị Hương", "bố", "mẹ", "sếp Cường", "cô Bích", "chú Hải",
    "bà ngoại", "em Vân", "anh hai", "chị Loan", "thầy Phúc",
]

STREETS_HN = [
    "Trần Duy Hưng", "Nguyễn Trãi", "Láng Hạ", "Kim Mã", "Cầu Giấy", "Xuân Thủy",
    "Phạm Hùng", "Lê Văn Lương", "Hoàng Quốc Việt", "Tây Sơn", "Giải Phóng",
    "Nguyễn Chí Thanh", "Thái Hà", "Bà Triệu", "Tôn Đức Thắng", "Đội Cấn",
    "Hoàng Hoa Thám", "Âu Cơ", "Nguyễn Văn Cừ", "Trần Đại Nghĩa", "Khâm Thiên",
]
STREETS_SG = [
    "Nguyễn Huệ", "Lê Lợi", "Điện Biên Phủ", "Cách Mạng Tháng Tám", "Võ Văn Tần",
    "Pasteur", "Nam Kỳ Khởi Nghĩa", "Trường Chinh", "Cộng Hòa", "Phan Xích Long",
    "Nguyễn Văn Linh", "Mai Chí Thọ", "Phạm Văn Đồng", "Lê Văn Sỹ", "Hoàng Văn Thụ",
]
PLACES = [
    "Hồ Gươm", "Lăng Bác", "Văn Miếu", "AEON Mall Long Biên", "Times City",
    "Vincom Bà Triệu", "sân bay Nội Bài", "ga Hà Nội", "Big C Thăng Long",
    "chợ Bến Thành", "Landmark 81", "Dinh Độc Lập", "sân bay Tân Sơn Nhất",
    "Vạn Hạnh Mall", "công viên Tao Đàn", "bến xe Mỹ Đình", "phố cổ Hội An",
    "Bà Nà Hills", "cầu Rồng Đà Nẵng", "Vincom Center Đồng Khởi",
]
EV = [
    "VinFast VF8", "VinFast VF9", "VinFast VF5", "VinFast VF e34", "VinFast VF6",
    "Tesla Model 3", "Tesla Model Y", "BYD Atto 3", "BYD Seal", "Hyundai Ioniq 5",
    "Kia EV6", "BMW iX3", "Mercedes EQS", "Audi e-tron", "Porsche Taycan", "Nissan Leaf",
]


def _sample(pool, k):
    return random.sample(pool, min(k, len(pool)))


def build_entity_hint(domains):
    """Vietnamese prompt block seeding concrete, varied slots for these domains.
    Returns "" when no pool is relevant."""
    lines, ds = [], set(domains)
    if "music" in ds:
        picks = _sample(SONGS, 8)
        lines.append("• bài hát (mỗi mẫu một bài KHÁC NHAU): "
                     + "; ".join(f"'{t}' - {a}" for t, a in picks))
    if ds & {"calling", "messaging"}:
        lines.append("• tên người liên hệ: " + ", ".join(_sample(NAMES, 7)))
    if ds & {"navigation", "charging"}:
        st = _sample(STREETS_HN, 3) + _sample(STREETS_SG, 3)
        lines.append("• địa chỉ: " + ", ".join(f"{random.randint(2, 300)} {s}" for s in st))
        lines.append("• địa điểm: " + ", ".join(_sample(PLACES, 4)))
    if ds & {"charging", "vehicle", "driver_assist"}:
        lines.append("• xe/EV: " + ", ".join(_sample(EV, 4)))
    if not lines:
        return ""
    return ("\n\nSLOT BẮT BUỘC — mỗi mẫu PHẢI dùng slot KHÁC NHAU lấy TỪ danh sách dưới đây, "
            "KHÔNG lặp trong batch, TUYỆT ĐỐI KHÔNG dùng slot ngoài danh sách "
            "(đặc biệt tránh các bài quá phổ biến như 'Nơi Này Có Anh', 'See Tình'):\n"
            + "\n".join(lines))
